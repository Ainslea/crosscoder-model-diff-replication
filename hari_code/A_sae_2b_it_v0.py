#!pip install sae-lens
#!pip install transformers bitsandbytes accelerate datasets tqdm matplotlib seaborn statsmodels

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os, json, re
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from statsmodels.stats.contingency_tables import mcnemar



# --------------------
# Setup device
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------
# CONFIG: 2B base vs 2B-IT
# --------------------
MODEL_A_ID = "google/gemma-2-2b"        # Base model (student / target)
MODEL_B_ID = "google/gemma-2-2b-it"     # 2B-IT model (teacher / source)

STITCH_LAYER_A = 20   # Gemma 2B layer 20
STITCH_LAYER_B = 20   # Gemma 2B-IT layer 20 (same index)

SAE_A_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_B_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_A_ID = f"layer_{STITCH_LAYER_A}/width_16k/canonical"
SAE_B_ID = f"layer_{STITCH_LAYER_B}/width_16k/canonical"

DATASET_NAME = "NeelNanda/openwebtext-tokenized-9b"
CONTEXT_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
ACTIVATION_CACHE_SIZE = 50_000       # number of sequences to cache for training
NUM_SAMPLES_FOR_SVCCA = 200

# GSM8K config
GSM8K_FEATURE_SAMPLES = 100
GSM8K_EVAL_SAMPLES = 500
NUM_TOP_FEATURES = 50
GRAFTING_STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_NEW_TOKENS = 400
EVAL_BATCH_SIZE = 8
CHECKPOINT_FILE = "grafting_checkpoint_2b_2bit.json"
EVALUATION_QUESTIONS_FILE = "gsm8k_eval_samples_2b_2bit.json"

# --------------------
# Hugging Face login (replace token string or use hf-cli)
# --------------------
from huggingface_hub import login



# --------------------
# Load tokenizer and models
# --------------------
print("Loading tokenizer and models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_A_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {MODEL_A_ID} (base)...")
model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print(f"Loading {MODEL_B_ID} (2B-IT)...")
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Models and tokenizer loaded successfully.")
print(f"Using fixed layer pair: 2B base Layer {STITCH_LAYER_A} ↔ 2B-IT Layer {STITCH_LAYER_B}")

# --------------------
# Helper functions
# --------------------
def get_unnormalized_activations(model, tokens, layer_idx, return_all_positions=False):
    with torch.no_grad():
        if hasattr(model, "model"):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)

        all_activations = outputs.hidden_states[layer_idx].to(torch.float32)

        if return_all_positions:
            b, seq, d = all_activations.shape
            return all_activations.view(-1, d).cpu()
        else:
            return all_activations[:, -1].cpu()

def get_normalized_activations(model, tokens, layer_idx, return_all_positions=False):
    with torch.no_grad():
        if hasattr(model, "model"):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)

        all_activations = outputs.hidden_states[layer_idx].to(torch.float32)

        if return_all_positions:
            b, seq, d = all_activations.shape
            reshaped = all_activations.view(-1, d)
            normalized = torch.nn.functional.layer_norm(reshaped, [d])
            return normalized.cpu()
        else:
            last_token_acts = all_activations[:, -1]
            normalized = torch.nn.functional.layer_norm(last_token_acts, [last_token_acts.shape[-1]])
            return normalized.cpu()

def get_sae_features(model, sae, tokens, layer_idx):
    """SAE features for last token"""
    with torch.no_grad():
        if hasattr(model, "model"):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)
        activations = outputs.hidden_states[layer_idx].to(torch.float32)
        features = sae.encode(activations[:, -1])
        return features.cpu()

def centered_kernel_alignment(acts_a, acts_b, device):
    acts_a = acts_a.to(device)
    acts_b = acts_b.to(device)

    acts_a = acts_a - acts_a.mean(dim=0, keepdim=True)
    acts_b = acts_b - acts_b.mean(dim=0, keepdim=True)

    gram_a = torch.mm(acts_a, acts_a.T)
    gram_b = torch.mm(acts_b, acts_b.T)

    n = gram_a.shape[0]
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    gram_a_centered = H @ gram_a @ H
    gram_b_centered = H @ gram_b @ H

    numerator = torch.trace(gram_a_centered @ gram_b_centered)
    denominator = torch.sqrt(
        torch.trace(gram_a_centered @ gram_a_centered) *
        torch.trace(gram_b_centered @ gram_b_centered)
    )

    if denominator > 1e-8:
        cka_score = (numerator / denominator).item()
    else:
        cka_score = 0.0

    return abs(cka_score)

class EnhancedStitch(nn.Module):
    def __init__(self, dim_a, dim_b, dropout_rate=0.1):
        super().__init__()
        self.up = nn.Linear(dim_a, dim_b)
        self.down = nn.Linear(dim_b, dim_a)
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.bias)

    def forward_up(self, x, use_dropout=True):   # A -> B
        x = self.up(x)
        if use_dropout and self.training:
            x = self.dropout(x)
        return x

    def forward_down(self, x, use_dropout=True): # B -> A
        x = self.down(x)
        if use_dropout and self.training:
            x = self.dropout(x)
        return x

class ActivationDataset(Dataset):
    def __init__(self, acts_a_tensor, acts_b_tensor):
        self.acts_a = acts_a_tensor
        self.acts_b = acts_b_tensor

    def __len__(self):
        return self.acts_a.shape[0]

    def __getitem__(self, idx):
        return self.acts_a[idx], self.acts_b[idx]



# --------------------
# Load SAEs
# --------------------
print("Loading Gemma Scope SAEs for evaluation...")
try:
    print(f"Loading SAE A (base): {SAE_A_RELEASE}/{SAE_A_ID}")
    sae_a: SAE = SAE.from_pretrained(
        release=SAE_A_RELEASE, sae_id=SAE_A_ID, device=device
    )

    print(f"Loading SAE B (2B-IT): {SAE_B_RELEASE}/{SAE_B_ID}")
    sae_b: SAE = SAE.from_pretrained(
        release=SAE_B_RELEASE, sae_id=SAE_B_ID, device=device
    )

    print("SAEs loaded successfully!")
    print(f"SAE A (Layer {STITCH_LAYER_A}): d_in={sae_a.cfg.d_in}, d_sae={sae_a.cfg.d_sae}")
    print(f"SAE B (Layer {STITCH_LAYER_B}): d_in={sae_b.cfg.d_in}, d_sae={sae_b.cfg.d_sae}")
except Exception as e:
    print(f"Warning: Could not load SAEs - {e}")
    sae_a, sae_b = None, None
    raise RuntimeError("SAEs are required for this experiment")



# ================================
# 1. CKA VERIFICATION (ACTIVATION SPACE)
# ================================
print(f"\nUsing fixed layer pair: 2B base Layer {STITCH_LAYER_A} ↔ 2B-IT Layer {STITCH_LAYER_B}")
print("Verifying layer pair with CKA similarity...")
print("Preparing small dataset for verification...")

verification_dataset = load_dataset(
    DATASET_NAME, split="train", streaming=True
).take(NUM_SAMPLES_FOR_SVCCA)

tokens_list = []
for item in verification_dataset:
    tokens = torch.tensor(item["tokens"])[:CONTEXT_LENGTH]
    if len(tokens) < CONTEXT_LENGTH:
        tokens = torch.cat(
            [tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)]
        )
    tokens_list.append(tokens.to(device))

all_acts_a, all_acts_b = [], []
for tokens in tqdm(tokens_list[:50], desc="Computing verification activations"):
    try:
        acts_a = get_normalized_activations(
            model_a, tokens.unsqueeze(0), STITCH_LAYER_A, return_all_positions=True
        )
        acts_b = get_normalized_activations(
            model_b, tokens.unsqueeze(0), STITCH_LAYER_B, return_all_positions=True
        )
        all_acts_a.append(acts_a)
        all_acts_b.append(acts_b)
    except Exception as e:
        print(f"Error in verification: {e}")
        continue

if len(all_acts_a) > 0:
    acts_a_stacked = torch.cat(all_acts_a, dim=0)
    acts_b_stacked = torch.cat(all_acts_b, dim=0)
    print(
        f"Verification using {acts_a_stacked.shape[0]} total positions "
        f"from {len(all_acts_a)} sequences"
    )

    cka_score = centered_kernel_alignment(acts_a_stacked, acts_b_stacked, device)
    print(
        f"CKA similarity between 2B base Layer {STITCH_LAYER_A} "
        f"and 2B-IT Layer {STITCH_LAYER_B}: {cka_score:.4f}"
    )
else:
    print("Could not compute verification - proceeding anyway")


#ACTIVATION_CACHE_SIZE = 50_000       # number of sequences to cache for training
ACTIVATION_CACHE_SIZE = 1000       # number of sequences to cache for training


# ================================
# 2. SAE FEATURE CACHING (LAST TOKEN)
# ================================
print("\nCaching SAE features for stitch training...")
train_dataset = load_dataset(
    DATASET_NAME, split="train", streaming=True
).take(ACTIVATION_CACHE_SIZE)

features_a_list, features_b_list = [], []
processed_count = 0

for item in tqdm(train_dataset, desc="Caching SAE features"):
    try:
        tokens = torch.tensor(item["tokens"])[:CONTEXT_LENGTH]
        if len(tokens) < CONTEXT_LENGTH:
            tokens = torch.cat(
                [tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)]
            )
        tokens = tokens.to(device).unsqueeze(0)

        feats_a = get_sae_features(model_a, sae_a, tokens, STITCH_LAYER_A)  # [1, d_sae_a]
        feats_b = get_sae_features(model_b, sae_b, tokens, STITCH_LAYER_B)  # [1, d_sae_b]

        features_a_list.append(feats_a.squeeze(0))
        features_b_list.append(feats_b.squeeze(0))

        processed_count += 1
        if processed_count % 100 == 0:
            torch.cuda.empty_cache()
            print(f"Cached {processed_count} sequences")

    except Exception as e:
        print(f"Error caching features: {e}")
        continue

print(f"Successfully cached {len(features_a_list)} SAE feature pairs")
torch.cuda.empty_cache()

cached_features_a_tensor = torch.stack(features_a_list, dim=0).to(torch.bfloat16)
cached_features_b_tensor = torch.stack(features_b_list, dim=0).to(torch.bfloat16)

print(
    f"Final training data shape (SAE space): "
    f"A={cached_features_a_tensor.shape}, B={cached_features_b_tensor.shape}"
)

activation_dataset = ActivationDataset(cached_features_a_tensor, cached_features_b_tensor)
dataloader = DataLoader(activation_dataset, batch_size=BATCH_SIZE, shuffle=True)



# ================================
# 3. STITCH INITIALIZATION (SAE SPACE)
# ================================
dim_a = sae_a.cfg.d_sae   # e.g. 16384
dim_b = sae_b.cfg.d_sae   # e.g. 16384

stitch_model = EnhancedStitch(dim_a, dim_b, dropout_rate=0.1).to(device)
optimizer = torch.optim.AdamW(stitch_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

def sae_loss_fn(pred_features, target_features, l1_weight=0.001):
    mse = F.mse_loss(pred_features, target_features)
    l1 = torch.abs(pred_features).mean()

    active_mask = (torch.abs(target_features) > 0.1).float()
    active_mse = (
        active_mask * (pred_features - target_features) ** 2
    ).sum() / (active_mask.sum() + 1e-8)

    false_negative = (active_mask * (1 - torch.sigmoid(pred_features * 10))).mean()

    return mse + (l1_weight * l1) + 0.3 * active_mse + 0.1 * false_negative

print(f"\nEnhanced Stitch Architecture (SAE space): {dim_a} -> {dim_b}")
print(f"Training data: {cached_features_a_tensor.shape[0]:,} SAE feature pairs")
print(f"Training for {NUM_EPOCHS} epochs with {len(dataloader)} batches per epoch")
print("Starting stitch training...")



# ================================
# 4. TRAINING LOOP (SAE FEATURES)
# ================================
stitch_model.train()
loss_history = []
epoch_losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_up_loss = 0.0
    total_down_loss = 0.0
    total_cycle_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch_idx, (f_a, f_b) in enumerate(progress_bar):
        f_a = f_a.to(device, dtype=torch.float32)
        f_b = f_b.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        f_a_to_b = stitch_model.forward_up(f_a)      # base -> IT
        f_b_to_a = stitch_model.forward_down(f_b)    # IT   -> base (grafting direction)

        f_a_cycle = stitch_model.forward_down(f_a_to_b, use_dropout=False)
        f_b_cycle = stitch_model.forward_up(f_b_to_a, use_dropout=False)

        up_loss = sae_loss_fn(f_a_to_b, f_b)
        down_loss = sae_loss_fn(f_b_to_a, f_a)
        cycle_a_loss = sae_loss_fn(f_a_cycle, f_a)
        cycle_b_loss = sae_loss_fn(f_b_cycle, f_b)

        loss = up_loss + 1.2 * down_loss + 0.5 * (cycle_a_loss + cycle_b_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(stitch_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_up_loss += up_loss.item()
        total_down_loss += down_loss.item()
        total_cycle_loss += (cycle_a_loss.item() + cycle_b_loss.item())

        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Base->IT": f"{up_loss.item():.4f}",
                "IT->Base": f"{down_loss.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

        if batch_idx % 100 == 0:
            loss_history.append(loss.item())

    avg_loss = total_loss / len(dataloader)
    avg_up_loss = total_up_loss / len(dataloader)
    avg_down_loss = total_down_loss / len(dataloader)
    avg_cycle_loss = total_cycle_loss / len(dataloader)

    epoch_losses.append(avg_loss)
    scheduler.step()

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Total Loss:       {avg_loss:.4f}")
    print(f"  Base->IT Loss:    {avg_up_loss:.4f}")
    print(f"  IT->Base Loss:    {avg_down_loss:.4f} (grafting direction)")
    print(f"  Cycle Loss (sum): {avg_cycle_loss:.4f}")
    print(f"  Learning Rate:    {scheduler.get_last_lr()[0]:.2e}")

print("Stitch training complete in SAE space!")



# ================================
# 5. PLOT TRAINING CURVES
# ================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, alpha=0.7, label="Batch Loss")
plt.title("Training Loss (Per Batch)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epoch_losses, "o-", linewidth=2, markersize=6)
plt.title("Average Loss Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"stitch_training_curves_SAE_2B-2B-IT_L{STITCH_LAYER_A}.png", dpi=150, bbox_inches="tight")
plt.show()



# ================================
# 6. SAE-SPACE EVALUATION (OpenWebText)
# ================================
print("\nEvaluating stitch quality in SAE space on held-out samples...")

test_samples = []
test_dataset = (
    load_dataset(DATASET_NAME, split="train", streaming=True)
    .skip(ACTIVATION_CACHE_SIZE // 4)
    .take(10)
)

for item in test_dataset:
    test_tokens = torch.tensor(item["tokens"])[:CONTEXT_LENGTH]
    if len(test_tokens) < CONTEXT_LENGTH:
        test_tokens = torch.cat(
            [test_tokens, torch.full((CONTEXT_LENGTH - len(test_tokens),), tokenizer.pad_token_id)]
        )
    test_samples.append(test_tokens.to(device).unsqueeze(0))

all_transfer_losses = []
all_random_losses = []
all_reverse_losses = []
all_feature_preservation_scores = []

stitch_model.eval()
model_a.eval()
model_b.eval()
sae_a.eval()
sae_b.eval()

for i, test_tokens in enumerate(test_samples):
    print(f"\nTest Sample {i+1}")
    try:
        with torch.no_grad():
            if hasattr(model_a, "model"):
                outputs_a = model_a.model(test_tokens, output_hidden_states=True)
            else:
                outputs_a = model_a(test_tokens, output_hidden_states=True)
            acts_a = outputs_a.hidden_states[STITCH_LAYER_A]

            if hasattr(model_b, "model"):
                outputs_b = model_b.model(test_tokens, output_hidden_states=True)
            else:
                outputs_b = model_b(test_tokens, output_hidden_states=True)
            acts_b = outputs_b.hidden_states[STITCH_LAYER_B]

            features_a_all = sae_a.encode(acts_a.squeeze(0))
            features_b_all = sae_b.encode(acts_b.squeeze(0))

    except Exception as e:
        print(f"Error: {e}")
        continue

    seq_len = features_a_all.shape[0]
    eval_positions = [0, seq_len // 2, seq_len - 1] if seq_len > 1 else [0]

    position_metrics = {
        "transfer": [],
        "reverse": [],
        "random": [],
        "feature_preservation": [],
    }

    for pos_idx in eval_positions:
        f_a = features_a_all[pos_idx].to(device)
        f_b = features_b_all[pos_idx].to(device)

        with torch.no_grad():
            f_a_to_b = stitch_model.forward_up(f_a, use_dropout=False)   # base->IT
            transfer_loss = F.mse_loss(f_a_to_b, f_b).item()

            f_b_to_a = stitch_model.forward_down(f_b, use_dropout=False) # IT->base
            reverse_loss = F.mse_loss(f_b_to_a, f_a).item()

            random_projection = nn.Linear(dim_b, dim_a).to(device)
            f_b_random = random_projection(f_b)
            random_loss = F.mse_loss(f_b_random, f_a).item()

            active_b = (torch.abs(f_b) > 0.1).float()
            active_transferred = (torch.abs(f_b_to_a) > 0.1).float()
            preserved = (active_b * active_transferred).sum()
            total_active_b = active_b.sum()
            preservation_rate = (preserved / total_active_b).item() if total_active_b > 0 else 0.0

            position_metrics["transfer"].append(transfer_loss)
            position_metrics["reverse"].append(reverse_loss)
            position_metrics["random"].append(random_loss)
            position_metrics["feature_preservation"].append(preservation_rate)

    avg_transfer = np.mean(position_metrics["transfer"])
    avg_reverse = np.mean(position_metrics["reverse"])
    avg_random = np.mean(position_metrics["random"])
    avg_preservation = np.mean(position_metrics["feature_preservation"])

    all_transfer_losses.append(avg_transfer)
    all_reverse_losses.append(avg_reverse)
    all_random_losses.append(avg_random)
    all_feature_preservation_scores.append(avg_preservation)

    print(f"  Base->IT Transfer MSE: {avg_transfer:.4f}")
    print(f"  IT->Base Transfer MSE: {avg_reverse:.4f} ← Grafting direction")
    print(f"  Random Baseline MSE:   {avg_random:.4f}")
    print(f"  Feature Preservation:  {avg_preservation:.2%}")

print("\n" + "="*60)
print("FINAL RESULTS (SAE Feature Space Transfer: 2B base ↔ 2B-IT)")
print("="*60)
print(f"Average Base->IT Transfer MSE:   {np.mean(all_transfer_losses):.4f}")
print(f"Average IT->Base Transfer MSE:   {np.mean(all_reverse_losses):.4f}")
print(f"Average Random Baseline MSE:     {np.mean(all_random_losses):.4f}")
print(f"Average Feature Preservation:    {np.mean(all_feature_preservation_scores):.2%}")
print("-" * 60)

up_improvement = np.mean(all_random_losses) / np.mean(all_transfer_losses)
down_improvement = np.mean(all_random_losses) / np.mean(all_reverse_losses)
print(f"Base->IT Improvement:            {up_improvement:.2f}x over random")
print(f"IT->Base Improvement:            {down_improvement:.2f}x over random")
print("="*60)



# ================================
# 7. (OPTIONAL) GSM8K REASONING + GRAFTING
# ================================
# Everything below still uses SAE-space stitch and 2B-IT -> 2B mapping,
# so you WON'T get any dimension mismatch here either.

def extract_numerical_answer(text):
    text = text.replace(",", "")
    patterns = [
        r"####\s*([+-]?\d+(?:\.\d+)?)",
        r"[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)",
        r"[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)",
        r"=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$)",
        r"\$([+-]?\d+(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(\.\d+)?)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            match = matches[-1]
            try:
                return float(match[0] if isinstance(match, tuple) else match)
            except (ValueError, IndexError):
                continue
    numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", text[-100:])
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None

def identify_reasoning_features(model_teacher, sae_teacher, tokenizer, gsm8k_samples, layer_idx):
    print(f"Identifying reasoning features using {len(gsm8k_samples)} GSM8K samples...")
    feature_activations = defaultdict(list)
    feature_frequency = defaultdict(int)
    model_teacher.eval()
    sae_teacher.eval()

    for i, sample in enumerate(tqdm(gsm8k_samples, desc="Processing GSM8K samples")):
        question = sample["question"]
        inputs = tokenizer(
            f"Question: {question}\nLet me solve this step by step:",
            return_tensors="pt",
            truncation=True,
            max_length=CONTEXT_LENGTH,
            padding=True,
        ).to(device)

        try:
            with torch.no_grad():
                if hasattr(model_teacher, "model"):
                    outputs = model_teacher.model(inputs["input_ids"], output_hidden_states=True)
                else:
                    outputs = model_teacher(inputs["input_ids"], output_hidden_states=True)

                all_activations = outputs.hidden_states[layer_idx]  # [1, seq_len, d]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
                masked_activations = all_activations * attention_mask
                seq_len = attention_mask.sum()
                avg_activation = masked_activations.sum(dim=1) / seq_len  # [1, d]

                feature_acts = sae_teacher.encode(avg_activation).squeeze(0).cpu().float()
                feature_magnitudes = torch.abs(feature_acts)

                top_indices = torch.topk(feature_magnitudes, NUM_TOP_FEATURES).indices

                for idx in top_indices:
                    fid = idx.item()
                    val = feature_acts[idx].item()
                    feature_activations[fid].append(val)
                    if abs(val) > 0.1:
                        feature_frequency[fid] += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    feature_scores = {}
    for fid, acts in feature_activations.items():
        if acts:
            avg_mag = np.mean(np.abs(acts))
            avg_signed = np.mean(acts)
            freq = feature_frequency[fid]
            score = freq * avg_mag
            feature_scores[fid] = {
                "score": score,
                "frequency": freq,
                "avg_activation": avg_signed,
                "avg_magnitude": avg_mag,
            }

    top_features = sorted(
        feature_scores.keys(), key=lambda x: feature_scores[x]["score"], reverse=True
    )[:NUM_TOP_FEATURES]

    print("\nTop 10 reasoning features (averaged across full sequences):")
    for i, fid in enumerate(top_features[:10]):
        info = feature_scores[fid]
        print(
            f"  Feature {fid}: score={info['score']:.3f}, "
            f"freq={info['frequency']}/{len(gsm8k_samples)}, "
            f"avg_act={info['avg_activation']:.4f}"
        )

    return top_features, feature_scores

grafting_context = {}



def offline_grafting_hook(module, args, output):
    original_activation = output[0] if isinstance(output, tuple) else output
    strength = grafting_context.get("strength", 0.0)
    if strength == 0.0:
        return output

    with torch.no_grad():
        reasoning_features_dict = grafting_context["reasoning_features_dict"]

        # create 2B-IT SAE feature vector
        f_B_reasoning = torch.zeros(1, sae_b.cfg.d_sae, device=device)

        feature_ids = list(reasoning_features_dict.keys())
        feature_values = torch.tensor(
            list(reasoning_features_dict.values()),
            device=device,
            dtype=f_B_reasoning.dtype,   # <<< IMPORTANT
        )

        f_B_reasoning[:, feature_ids] = feature_values

        # IT -> base in SAE space
        f_A_grafted = stitch_model.forward_down(f_B_reasoning, use_dropout=False)

        # decode to base residual stream
        h_A_grafted_unnorm = sae_a.decode(f_A_grafted)
        h_A_grafted_norm = F.layer_norm(
            h_A_grafted_unnorm, [h_A_grafted_unnorm.shape[-1]]
        )

        modified_activation = (
            (1 - strength) * original_activation + strength * h_A_grafted_norm
        )

    return (modified_activation,) + output[1:] if isinstance(output, tuple) else modified_activation




def evaluate_gsm8k(model, tokenizer, samples, strength=0.0):
    correct, total = 0, 0
    hook_handle = target_layer.register_forward_hook(offline_grafting_hook)
    print(f"\nEvaluating on GSM8K with grafting strength: {strength}")

    for i in tqdm(range(0, len(samples), EVAL_BATCH_SIZE), desc=f"Strength {strength}"):
        batch_samples = samples[i : i + EVAL_BATCH_SIZE]
        prompts = [
            f"Question: {s['question']}\nLet me solve this step by step:" for s in batch_samples
        ]
        correct_answers = [extract_numerical_answer(s["answer"]) for s in batch_samples]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        grafting_context["strength"] = strength
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for j, response in enumerate(responses):
            predicted_answer = extract_numerical_answer(response)
            ca = correct_answers[j]
            is_correct = (
                predicted_answer is not None
                and ca is not None
                and abs(predicted_answer - ca) < 0.01
            )
            if is_correct:
                correct += 1
            total += 1

    hook_handle.remove()
    accuracy = correct / total if total > 0 else 0.0
    print(f"-> Strength {strength}: Accuracy = {accuracy:.3f} ({correct}/{total})")
    return accuracy



# Much faster settings
GSM8K_FEATURE_SAMPLES = 30      # 100 -> 30
GSM8K_EVAL_SAMPLES = 100        # 500 -> 100
NUM_TOP_FEATURES = 20           # 50 -> 20 (less scoring work)
GRAFTING_STRENGTHS = [0.0, 0.3, 0.6, 1.0]  # fewer strengths
MAX_NEW_TOKENS = 128            # 400 -> 128 (shorter chains)
EVAL_BATCH_SIZE = 16            # 8 -> 16 (if your GPU fits)



# Load / create GSM8K eval slice
if os.path.exists(EVALUATION_QUESTIONS_FILE):
    print(f"\nLoading consistent evaluation samples from '{EVALUATION_QUESTIONS_FILE}'...")
    with open(EVALUATION_QUESTIONS_FILE, "r") as f:
        evaluation_samples = json.load(f)
    print(f"Loaded {len(evaluation_samples)} evaluation samples.")
else:
    print(f"\n'{EVALUATION_QUESTIONS_FILE}' not found. Generating and saving for future runs...")
    gsm8k_test = load_dataset("gsm8k", "main", split="test")
    evaluation_samples = list(gsm8k_test.select(range(GSM8K_EVAL_SAMPLES)))
    with open(EVALUATION_QUESTIONS_FILE, "w") as f:
        json.dump(evaluation_samples, f)
    print(f"Saved {len(evaluation_samples)} evaluation samples to '{EVALUATION_QUESTIONS_FILE}'.")

# Feature-identification split from GSM8K train
gsm8k_train = load_dataset("gsm8k", "main", split="train")
feature_id_samples = list(gsm8k_train.select(range(GSM8K_FEATURE_SAMPLES)))

# Load / resume grafting checkpoint
reasoning_features_dict = {}
results_by_strength = {}
if os.path.exists(CHECKPOINT_FILE):
    print(f"\nFound results checkpoint at '{CHECKPOINT_FILE}'. Loading progress...")
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_data = json.load(f)
    reasoning_features_dict = {int(k): v for k, v in checkpoint_data.get("reasoning_features_dict", {}).items()}
    results_by_strength = {float(k): v for k, v in checkpoint_data.get("results_by_strength", {}).items()}
    print("Progress loaded.")

# Identify reasoning features from 2B-IT if needed
if not reasoning_features_dict:
    print("\n" + "=" * 60)
    print("Identifying static reasoning features from 2B-IT (teacher)")
    print("=" * 60)
    top_reasoning_features_ids, feature_scores = identify_reasoning_features(
        model_b, sae_b, tokenizer, feature_id_samples, STITCH_LAYER_B
    )
    reasoning_features_dict = {
        fid: feature_scores[fid]["avg_activation"] for fid in top_reasoning_features_ids
    }

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {
                "reasoning_features_dict": reasoning_features_dict,
                "results_by_strength": results_by_strength,
            },
            f,
            indent=2,
        )
    print(f"Reasoning features identified and saved to '{CHECKPOINT_FILE}'.")
else:
    print("\n" + "=" * 60)
    print("Skipping feature identification (loaded from checkpoint)")
    print("=" * 60)

grafting_context["reasoning_features_dict"] = reasoning_features_dict
target_layer = model_a.model.layers[STITCH_LAYER_A]

print("\n" + "=" * 60)
print("Evaluating feature grafting (2B-IT → 2B base)")
print("=" * 60)

strengths_to_run = [s for s in GRAFTING_STRENGTHS if s not in results_by_strength]

if not strengths_to_run:
    print("All evaluation strengths are already complete and loaded from checkpoint.")
else:
    for strength in strengths_to_run:
        acc = evaluate_gsm8k(model_a, tokenizer, evaluation_samples, strength=strength)
        results_by_strength[strength] = acc
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(
                {
                    "reasoning_features_dict": reasoning_features_dict,
                    "results_by_strength": results_by_strength,
                },
                f,
                indent=2,
            )
        print(f"Completed and saved progress for strength {strength}.")

print("\n" + "=" * 60)
print("Analyzing grafting results")
print("=" * 60)

baseline_accuracy = results_by_strength.get(0.0, 0.0)
strengths = sorted(results_by_strength.keys())
accuracies = [results_by_strength[s] for s in strengths]

plt.figure(figsize=(12, 7))
plt.plot(strengths, accuracies, "o-", linewidth=2, markersize=8)
plt.xlabel("Grafting Strength (Additive Factor)")
plt.ylabel("GSM8K Accuracy")
plt.title("Offline Targeted Feature Grafting (2B-IT → 2B base)")
plt.grid(True, alpha=0.3)
plt.axhline(
    y=baseline_accuracy,
    linestyle="--",
    label=f"Baseline (strength 0.0): {baseline_accuracy:.3f}",
)
if len(accuracies) > 1:
    best_accuracy = max(accuracies)
    best_strength = strengths[int(np.argmax(accuracies))]
    plt.axvline(
        x=best_strength,
        linestyle="--",
        alpha=0.7,
        label=f"Best: {best_strength} ({best_accuracy:.3f})",
    )
plt.legend()
plt.tight_layout()
plt.savefig("gsm8k_offline_grafting_results_2b_2bit.png", dpi=150)
plt.show()

print("\nSummary")
for strength in sorted(results_by_strength.keys()):
    accuracy = results_by_strength[strength]
    improvement = (
        ((accuracy - baseline_accuracy) / baseline_accuracy * 100)
        if baseline_accuracy > 0
        else float("inf")
    )
    print(
        f"Strength {strength:3.1f}: Accuracy = {accuracy:.3f}  |  Improvement = {improvement:+.2f}%"
    )

new_strengths = strengths
new_accuracies = accuracies

plt.figure(figsize=(10, 6))
plt.plot(np.array(new_strengths), np.array(new_accuracies) * len(evaluation_samples),
         "o-", linewidth=2, markersize=8)
plt.xlabel("Grafting Strength")
plt.ylabel(f"GSM8K Questions Answered Correctly (out of {len(evaluation_samples)})")
plt.title("Feature Grafting Performance vs. Grafting Strength (2B-IT → 2B base)")
plt.grid(True, alpha=0.3)
plt.axhline(
    y=new_accuracies[0] * len(evaluation_samples),
    linestyle="--",
    label=f"Baseline (2B): {new_accuracies[0] * len(evaluation_samples):.0f}",
)

best_strength = new_strengths[int(np.argmax(new_accuracies))]
best_accuracy = max(new_accuracies)
plt.axvline(
    x=best_strength,
    linestyle="--",
    alpha=0.7,
    label=f"Best: {best_strength} ({best_accuracy * len(evaluation_samples):.0f})",
)
plt.legend()
plt.tight_layout()
plt.savefig("gsm8k_feature_grafting_results_2b_2bit.png", dpi=150, bbox_inches="tight")
plt.show()
