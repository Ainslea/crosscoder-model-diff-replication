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

# KL-based stitch training config
NUM_EPOCHS = 1                    # effectively controls LR schedule only
LEARNING_RATE = 1e-4
OPENWEBTEXT_TRAIN_STEPS = 500     # [REVISION(2)] number of KL steps on OWT

NUM_SAMPLES_FOR_SVCCA = 200

# GSM8K config
GSM8K_FEATURE_SAMPLES = 100
GSM8K_EVAL_SAMPLES = 500
NUM_TOP_FEATURES = 50

# Grafting strengths to sweep
GRAFTING_STRENGTHS = [0.0, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 1e-1, 2e-1, 3e-1]
MAX_NEW_TOKENS = 400
EVAL_BATCH_SIZE = 8
CHECKPOINT_FILE = "grafting_checkpoint_2b_2bit.json"
EVALUATION_QUESTIONS_FILE = "gsm8k_eval_samples_2b_2bit.json"

# Target norm for teacher SAE features before stitching
TARGET_SAE_NORM = 10.0

# --------------------
# Hugging Face login
# --------------------
from huggingface_hub import login
login(token="XXX")  # <--- REPLACE WITH YOUR TOKEN OR USE CLI

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
    low_cpu_mem_usage=True,
)

print(f"Loading {MODEL_B_ID} (2B-IT)...")
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("Models and tokenizer loaded successfully.")
print(f"Using fixed layer pair: 2B base Layer {STITCH_LAYER_A} ↔ 2B-IT Layer {STITCH_LAYER_B}")

# We’ll often need the underlying transformer stack
base_stack_a = model_a.model
base_stack_b = model_b.model

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
    """
    Simple SAE-space stitch: A_SAE <-> B_SAE via 2 linear layers.
    We train it functionally (KL on logits) rather than pure L2 on features.
    """
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

# --------------------
# CKA VERIFICATION
# --------------------
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

# --------------------
# Initialize stitch in SAE space
# --------------------
dim_a = sae_a.cfg.d_sae   # e.g. 16384
dim_b = sae_b.cfg.d_sae   # e.g. 16384

stitch_model = EnhancedStitch(dim_a, dim_b, dropout_rate=0.1).to(device)
optimizer = torch.optim.AdamW(stitch_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"\nEnhanced Stitch Architecture (SAE space): {dim_a} -> {dim_b}")
print("Training objective: KL(Teacher logits || Student+Stitch logits) [REVISION(2)]")
print(f"Using up to {OPENWEBTEXT_TRAIN_STEPS} training steps from OpenWebText")

# --------------------
# Grafting context + hooks
# --------------------
grafting_context = {}
target_layer = base_stack_a.layers[STITCH_LAYER_A]  # student layer to graft into

# [REVISION(3)] Training-time grafting hook (additive)
def training_grafting_hook(module, args, output):
    """
    Forward hook used *only during KL-training* of the stitch.

    Uses teacher SAE features from grafting_context["teacher_features_train"]
    and applies an ADDITIVE intervention to the student's last-token residual
    at STITCH_LAYER_A:

        r_student_last <- r_student_last + alpha * graft

    where alpha = grafting_context["train_strength"] (typically 1.0).
    """
    original_activation = output[0] if isinstance(output, tuple) else output  # [B, T, D]
    strength = grafting_context.get("train_strength", 1.0)

    if strength == 0.0 or "teacher_features_train" not in grafting_context:
        return output

    with torch.no_grad():
        device_local = original_activation.device

        # Teacher SAE features for this batch (B, d_sae_B)
        teacher_sae = grafting_context["teacher_features_train"].to(device_local)

        # Dtypes
        stitch_weight_dtype = stitch_model.down.weight.dtype     # usually float32
        sae_a_dtype = next(sae_a.parameters()).dtype             # usually float32
        resid_dtype = original_activation.dtype                  # e.g. bfloat16

        # 1) Ensure teacher SAE uses stitch dtype
        f_B = teacher_sae.to(dtype=stitch_weight_dtype)

        # 2) Map B -> A in SAE space (teacher -> student SAE space)
        f_A = stitch_model.forward_down(f_B, use_dropout=False)  # [B, d_sae_A], float32

        # 3) Decode back into student residual space via SAE-A
        h_A_unnorm = sae_a.decode(f_A.to(sae_a_dtype))           # [B, D_model]

        # 4) LayerNorm in SAE-A dtype, then cast to residual dtype/device
        h_A_norm = F.layer_norm(h_A_unnorm, [h_A_unnorm.shape[-1]])

        # Important: cast in two steps to avoid .to() signature issues
        h_A_norm = h_A_norm.to(device_local)
        h_A_norm = h_A_norm.to(resid_dtype)

        # 5) ADDITIVE graft ONLY on last token  [REVISION(3)]
        B, T, D = original_activation.shape
        modified_activation = original_activation.clone()

        last_orig = original_activation[:, -1, :]               # [B, D]
        last_grafted = last_orig + strength * h_A_norm          # ADDITIVE intervention
        modified_activation[:, -1, :] = last_grafted

    if isinstance(output, tuple):
        return (modified_activation,) + output[1:]
    else:
        return modified_activation

# --------------------
# KL-based stitch training step [REVISION(2)]
# --------------------
def kl_train_step(batch_items):
    """
    One training step on a small batch of OpenWebText tokens.
    Objective: minimize KL(Teacher || Student+Stitch).

    Robust to teacher outputs that come back as BaseModelOutputWithPast by
    reconstructing logits via lm_head if needed.
    """
    # Build tensor batch
    batch_tokens = []
    for item in batch_items:
        tokens = torch.tensor(item["tokens"])[:CONTEXT_LENGTH]
        if len(tokens) < CONTEXT_LENGTH:
            tokens = torch.cat(
                [tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)]
            )
        batch_tokens.append(tokens)

    input_ids = torch.stack(batch_tokens, dim=0).to(device)  # [B, T]
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # 1) Teacher forward (CausalLM) with hidden states
    with torch.no_grad():
        outputs_teacher = model_b(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get logits robustly
        if hasattr(outputs_teacher, "logits") and outputs_teacher.logits is not None:
            teacher_logits = outputs_teacher.logits  # [B, T, V]
        else:
            # Fallback: compute logits from last_hidden_state via lm_head
            if hasattr(outputs_teacher, "last_hidden_state"):
                hidden_for_logits = outputs_teacher.last_hidden_state  # [B, T, D]
            else:
                hidden_for_logits = outputs_teacher.hidden_states[-1]
            teacher_logits = model_b.lm_head(hidden_for_logits)

        # Hidden states at stitch layer for SAE-B
        if outputs_teacher.hidden_states is not None:
            teacher_hidden = outputs_teacher.hidden_states[STITCH_LAYER_B]  # [B, T, D_model]
        else:
            base_out = base_stack_b(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_hidden = base_out.hidden_states[STITCH_LAYER_B]

        # Average teacher hidden across non-pad tokens for SAE-B encoding
        attn = attention_mask.unsqueeze(-1)  # [B, T, 1]
        masked = teacher_hidden * attn
        lengths = attn.squeeze(-1).sum(dim=1, keepdim=True)  # [B, 1]
        lengths = torch.clamp(lengths, min=1.0)
        avg_act = masked.sum(dim=1) / lengths  # [B, D_model]

        teacher_sae = sae_b.encode(avg_act.to(torch.float32))  # [B, d_sae_B], float32

    # Store teacher SAE features for training hook
    grafting_context["teacher_features_train"] = teacher_sae.detach()
    grafting_context["train_strength"] = 1.0  # strong internal signal; external sweep is separate

    # 2) Student forward + graft (CausalLM)
    outputs_student = model_a(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    student_logits = outputs_student.logits  # [B, T, V]

    # 3) KL(Teacher || Student+Stitch) across non-pad tokens
    teacher_logits_use = teacher_logits[:, :-1, :]        # [B, T-1, V]
    student_logits_use = student_logits[:, :-1, :]
    mask = attention_mask[:, 1:].unsqueeze(-1)           # [B, T-1, 1]

    teacher_log_probs = F.log_softmax(teacher_logits_use.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits_use.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    kl_per_token_vocab = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    )  # [B, T-1, V]
    kl_per_token = kl_per_token_vocab.sum(dim=-1)  # [B, T-1]

    masked_kl = kl_per_token * mask.squeeze(-1)
    loss = masked_kl.sum() / (mask.sum() + 1e-8)

    return loss

# --------------------
# Run KL-based stitch training
# --------------------
print("\nStarting functional stitch training (KL-based)... [REVISION(2)]")

# Clear any existing hooks and attach training hook
try:
    target_layer._forward_hooks.clear()
    target_layer._forward_pre_hooks.clear()
except Exception as e:
    print("Warning while clearing hooks:", e)

train_hook_handle = target_layer.register_forward_hook(training_grafting_hook)

openwebtext_stream = load_dataset(
    DATASET_NAME, split="train", streaming=True
).take(OPENWEBTEXT_TRAIN_STEPS * BATCH_SIZE)

batch = []
step = 0
loss_history = []

for item in openwebtext_stream:
    batch.append(item)
    if len(batch) == BATCH_SIZE:
        batch_items = batch
        batch = []

        stitch_model.train()
        optimizer.zero_grad()
        loss = kl_train_step(batch_items)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stitch_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        loss_history.append(loss.item())
        if step % 10 == 0:
            print(f"[KL step {step}] loss = {loss.item():.4f}")
        if step >= OPENWEBTEXT_TRAIN_STEPS:
            break

train_hook_handle.remove()
print("KL-based stitch training complete.\n")

# ================================
# 6. GSM8K REASONING + DYNAMIC GRAFTING
# ================================
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

# [REVISION(1: still freq+magnitude, but with clear scoring)]
def identify_reasoning_features(model_teacher, sae_teacher, tokenizer, gsm8k_samples, layer_idx):
    """
    Heuristic feature selection: frequency × magnitude over GSM8K prompts.
    (Keeps original heuristic but wrapped more cleanly; ablation-based
    scoring could be plugged in here later.)
    """
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
                outputs = model_teacher.model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                all_activations = outputs.hidden_states[layer_idx]  # [1, seq_len, D]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
                masked_activations = all_activations * attention_mask
                seq_len = attention_mask.sum()
                avg_activation = masked_activations.sum(dim=1) / seq_len  # [1, D]

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

    print("\nTop 10 reasoning features (heuristic):")
    for i, fid in enumerate(top_features[:10]):
        info = feature_scores[fid]
        print(
            f"  Feature {fid}: score={info['score']:.3f}, "
            f"freq={info['frequency']}/{len(gsm8k_samples)}, "
            f"avg_act={info['avg_activation']:.4f}"
        )

    return top_features, feature_scores

# [REVISION(3)] Offline grafting hook for evaluation (additive, not convex)
def offline_grafting_hook(module, args, output):
    """
    Forward hook on Gemma 2B base model at STITCH_LAYER_A for GSM8K eval.

    Uses dynamic teacher SAE features per batch, stored in
    grafting_context["teacher_features"] by evaluate_gsm8k.

    Grafting is applied ONLY on the last token at this layer via ADDITION:
        r_student_last <- r_student_last + strength * graft
    """
    original_activation = output[0] if isinstance(output, tuple) else output  # [B, T, D]
    strength = grafting_context.get("strength", 0.0)

    if strength == 0.0 or "teacher_features" not in grafting_context:
        return output

    with torch.no_grad():
        device_local = original_activation.device

        teacher_sae = grafting_context["teacher_features"].to(device_local)
        if teacher_sae.shape[0] != original_activation.shape[0]:
            return output  # shape mismatch safety

        stitch_weight_dtype = stitch_model.down.weight.dtype
        sae_a_dtype = next(sae_a.parameters()).dtype
        resid_dtype = original_activation.dtype

        f_B = teacher_sae.to(dtype=stitch_weight_dtype)
        f_A = stitch_model.forward_down(f_B, use_dropout=False)  # [B, d_sae_A]

        h_A_unnorm = sae_a.decode(f_A.to(dtype=sae_a_dtype))     # [B, D_model]

        h_A_norm = F.layer_norm(h_A_unnorm, [h_A_unnorm.shape[-1]])
        h_A_norm = h_A_norm.to(device_local)
        h_A_norm = h_A_norm.to(resid_dtype)

        B, T, D = original_activation.shape
        modified_activation = original_activation.clone()

        last_orig = original_activation[:, -1, :]
        last_grafted = last_orig + strength * h_A_norm
        modified_activation[:, -1, :] = last_grafted

    if isinstance(output, tuple):
        return (modified_activation,) + output[1:]
    else:
        return modified_activation

def evaluate_gsm8k(model, tokenizer, samples, strength=0.0, use_random_features=False):
    """
    Evaluate GSM8K with dynamic teacher-based grafting.

    For each batch:
      1) Run teacher model (2B-IT) on the *same prompts*.
      2) Get average hidden states at STITCH_LAYER_B.
      3) Encode with SAE-B -> teacher_sae [B, d_sae_B].
      4) Keep either top-k reasoning features OR k random features.  [REVISION(4)]
      5) L2-normalize to TARGET_SAE_NORM.
      6) Store in grafting_context["teacher_features"].
      7) Run base model with grafting hook active.
    """
    correct, total = 0, 0
    # Attach evaluation hook
    eval_hook_handle = target_layer.register_forward_hook(offline_grafting_hook)
    print(
        f"\nEvaluating on GSM8K with grafting strength: {strength} "
        f"({'RANDOM' if use_random_features else 'REASONING'})"
    )

    model_b.eval()
    sae_b.eval()

    feature_ids_tensor = None
    if not use_random_features and grafting_context.get("reasoning_features_dict", None):
        feature_ids = list(grafting_context["reasoning_features_dict"].keys())
        feature_ids_tensor = torch.tensor(feature_ids, dtype=torch.long, device=device)
    elif use_random_features:
        # random control: same k as reasoning features
        k = len(grafting_context.get("reasoning_features_dict", {}))
        if k > 0:
            all_ids = torch.arange(sae_b.cfg.d_sae, device=device)
            perm = torch.randperm(sae_b.cfg.d_sae, device=device)
            feature_ids_tensor = perm[:k]
        else:
            feature_ids_tensor = None

    for i in tqdm(range(0, len(samples), EVAL_BATCH_SIZE),
                  desc=f"Strength {strength} ({'rand' if use_random_features else 'sel'})"):
        batch_samples = samples[i : i + EVAL_BATCH_SIZE]
        prompts = [
            f"Question: {s['question']}\nLet me solve this step by step:"
            for s in batch_samples
        ]
        correct_answers = [extract_numerical_answer(s["answer"]) for s in batch_samples]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Compute dynamic teacher SAE features per batch
        with torch.no_grad():
            outputs_teacher = model_b.model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            teacher_hidden = outputs_teacher.hidden_states[STITCH_LAYER_B]  # [B, T, D]
            attn = inputs["attention_mask"].unsqueeze(-1)                   # [B, T, 1]
            masked = teacher_hidden * attn
            lengths = attn.squeeze(-1).sum(dim=1, keepdim=True)             # [B, 1]
            lengths = torch.clamp(lengths, min=1.0)
            avg_act = masked.sum(dim=1) / lengths                           # [B, D]

            teacher_sae = sae_b.encode(avg_act.to(torch.float32))           # [B, d_sae_B]
            teacher_sae = teacher_sae.to(torch.float32)

            # Keep only selected or random feature dims (mask others to zero)
            if feature_ids_tensor is not None:
                mask = torch.zeros_like(teacher_sae)
                mask[:, feature_ids_tensor] = 1.0
                teacher_sae = teacher_sae * mask

            # Normalize to TARGET_SAE_NORM
            norms = teacher_sae.norm(dim=-1, keepdim=True)                  # [B, 1]
            target = TARGET_SAE_NORM
            teacher_sae = torch.where(
                norms > 0,
                teacher_sae / norms * target,
                teacher_sae,
            )

        grafting_context["teacher_features"] = teacher_sae
        grafting_context["strength"] = strength

        # Debug once if you want
        if grafting_context.get("debug_once_eval", True):
            print("[grafting eval] teacher_sae norm (mean):", norms.mean().item())
            grafting_context["debug_once_eval"] = False

        # Run base model with graft
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

    eval_hook_handle.remove()
    accuracy = correct / total if total > 0 else 0.0
    print(f"-> Strength {strength} ({'RANDOM' if use_random_features else 'REASONING'}): "
          f"Accuracy = {accuracy:.3f} ({correct}/{total})")
    return accuracy

# ================================
# GSM8K DATA & FEATURE IDENTIFICATION
# ================================
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
results_by_strength_random = {}   # [REVISION(4)] random control
if os.path.exists(CHECKPOINT_FILE):
    print(f"\nFound results checkpoint at '{CHECKPOINT_FILE}'. Loading progress...")
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_data = json.load(f)
    reasoning_features_dict = {
        int(k): v for k, v in checkpoint_data.get("reasoning_features_dict", {}).items()
    }
    results_by_strength = {
        float(k): v for k, v in checkpoint_data.get("results_by_strength", {}).items()
    }
    # support older checkpoints that lack random results
    if "results_by_strength_random" in checkpoint_data:
        results_by_strength_random = {
            float(k): v for k, v in checkpoint_data.get("results_by_strength_random", {}).items()
        }
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
                "results_by_strength_random": results_by_strength_random,
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

print("\n" + "=" * 60)
print("Evaluating feature grafting (2B-IT → 2B base) with dynamic teacher features")
print("=" * 60)

strengths_to_run = [s for s in GRAFTING_STRENGTHS if s not in results_by_strength]

if not strengths_to_run:
    print("All evaluation strengths are already complete and loaded from checkpoint.")
else:
    for strength in strengths_to_run:
        acc_sel = evaluate_gsm8k(model_a, tokenizer, evaluation_samples, strength=strength,
                                 use_random_features=False)
        acc_rand = evaluate_gsm8k(model_a, tokenizer, evaluation_samples, strength=strength,
                                  use_random_features=True)  # [REVISION(4)]

        results_by_strength[strength] = acc_sel
        results_by_strength_random[strength] = acc_rand

        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(
                {
                    "reasoning_features_dict": reasoning_features_dict,
                    "results_by_strength": results_by_strength,
                    "results_by_strength_random": results_by_strength_random,
                },
                f,
                indent=2,
            )
        print(f"Completed and saved progress for strength {strength}.")

print("\n" + "=" * 60)
print("Analyzing grafting results (selected vs random features) [REVISION(4)]")
print("=" * 60)

baseline_accuracy = results_by_strength.get(0.0, 0.0)
baseline_random = results_by_strength_random.get(0.0, 0.0)
strengths = sorted(results_by_strength.keys())
accuracies_sel = [results_by_strength[s] for s in strengths]
accuracies_rand = [results_by_strength_random.get(s, 0.0) for s in strengths]

plt.figure(figsize=(12, 7))
plt.plot(strengths, accuracies_sel, "o-", linewidth=2, markersize=8, label="Selected features")
plt.plot(strengths, accuracies_rand, "s--", linewidth=2, markersize=6, label="Random features")
plt.xlabel("Grafting Strength (Additive Factor)")
plt.ylabel("GSM8K Accuracy")
plt.title("Offline Targeted Feature Grafting: Selected vs Random Features")
plt.grid(True, alpha=0.3)
plt.axhline(
    y=baseline_accuracy,
    linestyle="--",
    label=f"Baseline (2B, selected) at 0.0: {baseline_accuracy:.3f}",
)
plt.axhline(
    y=baseline_random,
    linestyle=":",
    label=f"Baseline (2B, random) at 0.0: {baseline_random:.3f}",
)
if len(accuracies_sel) > 1:
    best_accuracy = max(accuracies_sel)
    best_strength = strengths[int(np.argmax(accuracies_sel))]
    plt.axvline(
        x=best_strength,
        linestyle="--",
        alpha=0.7,
        label=f"Best selected: {best_strength} ({best_accuracy:.3f})",
    )
plt.legend()
plt.tight_layout()
plt.savefig("gsm8k_offline_grafting_results_selected_vs_random.png", dpi=150)
plt.show()

print("\nSummary (Selected vs Random)")
for strength in strengths:
    acc_sel = results_by_strength[strength]
    acc_rand = results_by_strength_random.get(strength, 0.0)
    improvement_sel = (
        ((acc_sel - baseline_accuracy) / baseline_accuracy * 100)
        if baseline_accuracy > 0
        else float("inf")
    )
    improvement_rand = (
        ((acc_rand - baseline_random) / baseline_random * 100)
        if baseline_random > 0
        else float("inf")
    )
    print(
        f"Strength {strength:3.2f}: "
        f"Selected = {acc_sel:.3f} ({improvement_sel:+.2f}%) | "
        f"Random = {acc_rand:.3f} ({improvement_rand:+.2f}%)"
    )

# ============================================
# BASELINE EVALUATION: 2B base vs 2B-IT (no grafting)
# ============================================
def evaluate_gsm8k_plain(model, tokenizer, samples, tag="",
                         max_new_tokens=MAX_NEW_TOKENS,
                         batch_size=EVAL_BATCH_SIZE):
    """
    Plain GSM8K evaluation with no grafting or hooks.
    Uses the same extract_numerical_answer + prompts as grafting code.
    """
    model.eval()
    correct, total = 0, 0

    print(f"\n[BASELINE] Evaluating {tag} on GSM8K (no grafting)...")

    for i in tqdm(range(0, len(samples), batch_size), desc=f"Baseline {tag}"):
        batch_samples = samples[i : i + batch_size]
        prompts = [
            f"Question: {s['question']}\nLet me solve this step by step:"
            for s in batch_samples
        ]
        correct_answers = [extract_numerical_answer(s["answer"]) for s in batch_samples]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        responses = tokenizer.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True
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

    accuracy = correct / total if total > 0 else 0.0
    print(f"[BASELINE] {tag}: Accuracy = {accuracy:.3f} ({correct}/{total})")
    return accuracy

baseline_2b_base = evaluate_gsm8k_plain(
    model_a, tokenizer, evaluation_samples, tag="Gemma 2B base"
)
baseline_2b_it = evaluate_gsm8k_plain(
    model_b, tokenizer, evaluation_samples, tag="Gemma 2B-IT (teacher)"
)

print("\n===== BASELINE SUMMARY (NO GRAFTING) =====")
print(f"Gemma 2B base accuracy:      {baseline_2b_base:.3f}")
print(f"Gemma 2B-IT (teacher) acc.:  {baseline_2b_it:.3f}")

print("\nComparison vs grafting runs (selected features):")
print("Strength | Accuracy | Δ vs plain 2B base")
for s in sorted(results_by_strength.keys()):
    acc = results_by_strength[s]
    delta = acc - baseline_2b_base
    print(f"{s:7.3f} | {acc:8.3f} | {delta:+.3f}")
