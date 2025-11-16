import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_A_ID = "google/gemma-2-2b"     
MODEL_B_ID = "google/gemma-2-9b"    

# Fixed layer pair based on your SVCCA calculation
STITCH_LAYER_A = 8   # Gemma 2B layer 8
STITCH_LAYER_B = 17  # Gemma 9B layer 17

SAE_A_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_A_ID = f"layer_{STITCH_LAYER_A}/width_16k/canonical"  
SAE_B_RELEASE = "gemma-scope-9b-pt-res-canonical" 
SAE_B_ID = f"layer_{STITCH_LAYER_B}/width_16k/canonical" 
DATASET_NAME = "NeelNanda/openwebtext-tokenized-9b"
CONTEXT_LENGTH = 128
BATCH_SIZE = 32  # Increased batch size for better training
NUM_EPOCHS = 8   # Increased epochs for better convergence
LEARNING_RATE = 1e-4
ACTIVATION_CACHE_SIZE = 50_000  # Increased for more diverse training data
NUM_SAMPLES_FOR_SVCCA = 200  

print("Loading tokenizer and models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_A_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model A (2B model)
print(f"Loading {MODEL_A_ID}...")
model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Load Model B (9B model)
print(f"Loading {MODEL_B_ID}...")
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B_ID,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("Models and tokenizer loaded successfully.")
print(f"Using fixed layer pair: Gemma 2B Layer {STITCH_LAYER_A} → Gemma 9B Layer {STITCH_LAYER_B}")

def get_unnormalized_activations(model, tokens, layer_idx, return_all_positions=False):
    """
    Get unnormalized activations. Can return all positions or just the last token.
    """
    with torch.no_grad():
        # Handle different model architectures
        if hasattr(model, 'model'):
            # For Gemma models
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)
        
        # Shape: [batch_size, seq_len, hidden_dim]
        all_activations = outputs.hidden_states[layer_idx].to(torch.float32)
        
        if return_all_positions:
            # Return all sequence positions: [batch_size * seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = all_activations.shape
            return all_activations.view(-1, hidden_dim).cpu()
        else:
            # Return only last token: [batch_size, hidden_dim] 
            return all_activations[:, -1].cpu()

def get_normalized_activations(model, tokens, layer_idx, return_all_positions=False):
    """
    Get normalized activations. Can return all positions or just the last token.
    """
    with torch.no_grad():
        if hasattr(model, 'model'):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)
            
        # Shape: [batch_size, seq_len, hidden_dim]
        all_activations = outputs.hidden_states[layer_idx].to(torch.float32)
        
        if return_all_positions:
            # Normalize per position, return all: [batch_size * seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = all_activations.shape
            # Reshape to process all positions at once
            reshaped = all_activations.view(-1, hidden_dim)
            normalized = torch.nn.functional.layer_norm(reshaped, [hidden_dim])
            return normalized.cpu()
        else:
            # Normalize and return only last token: [batch_size, hidden_dim]
            last_token_acts = all_activations[:, -1]  
            normalized_activations = torch.nn.functional.layer_norm(last_token_acts, [last_token_acts.shape[-1]])
            return normalized_activations.cpu()

def centered_kernel_alignment(acts_a, acts_b, device):
    """
    Centered Kernel Alignment (CKA) for similarity measurement
    """
    acts_a = acts_a.to(device)
    acts_b = acts_b.to(device)
    
    # Center the data
    acts_a = acts_a - acts_a.mean(dim=0, keepdim=True)
    acts_b = acts_b - acts_b.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    gram_a = torch.mm(acts_a, acts_a.T)
    gram_b = torch.mm(acts_b, acts_b.T)
    
    # Center the Gram matrices
    n = gram_a.shape[0]
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    gram_a_centered = torch.mm(torch.mm(H, gram_a), H)
    gram_b_centered = torch.mm(torch.mm(H, gram_b), H)
    
    # CKA formula
    numerator = torch.trace(torch.mm(gram_a_centered, gram_b_centered))
    denominator = torch.sqrt(torch.trace(torch.mm(gram_a_centered, gram_a_centered)) * 
                           torch.trace(torch.mm(gram_b_centered, gram_b_centered)))
    
    if denominator > 1e-8:
        cka_score = (numerator / denominator).item()
    else:
        cka_score = 0.0
    
    return abs(cka_score)

# Verify the layer pair makes sense by computing CKA
print("Verifying layer pair with CKA similarity...")
print("Preparing small dataset for verification...")
verification_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).take(NUM_SAMPLES_FOR_SVCCA)

tokens_list = []
for item in verification_dataset:
    tokens = torch.tensor(item['tokens'])[:CONTEXT_LENGTH]
    if len(tokens) < CONTEXT_LENGTH:
        tokens = torch.cat([tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)])
    tokens_list.append(tokens.to(device))

# Compute activations for verification - using all positions for better statistics
all_acts_a, all_acts_b = [], []
for tokens in tqdm(tokens_list[:50], desc="Computing verification activations"):  # Reduced due to more data per sample
    try:
        # Get activations from ALL sequence positions
        act_a = get_normalized_activations(model_a, tokens.unsqueeze(0), STITCH_LAYER_A, return_all_positions=True)
        act_b = get_normalized_activations(model_b, tokens.unsqueeze(0), STITCH_LAYER_B, return_all_positions=True)
        all_acts_a.append(act_a)
        all_acts_b.append(act_b)
    except Exception as e:
        print(f"Error in verification: {e}")
        continue

if len(all_acts_a) > 0:
    # Concatenate all activations (each item now contains multiple positions)
    acts_a_stacked = torch.cat(all_acts_a, dim=0)  # [total_positions, dim_a]
    acts_b_stacked = torch.cat(all_acts_b, dim=0)  # [total_positions, dim_b]
    print(f"Verification using {acts_a_stacked.shape[0]} total positions from {len(all_acts_a)} sequences")
    
    cka_score = centered_kernel_alignment(acts_a_stacked, acts_b_stacked, device)
    print(f"CKA similarity between Layer {STITCH_LAYER_A} and Layer {STITCH_LAYER_B}: {cka_score:.4f}")
    
    if cka_score > 0.1:
        print("Good layer pair - proceeding with training")
    else:
        print("Low similarity - but proceeding anyway")
else:
    print("Could not compute verification - proceeding anyway")

class EnhancedStitch(nn.Module):
    """
    Enhanced stitch with better initialization and optional regularization
    """
    def __init__(self, dim_a, dim_b, dropout_rate=0.1):
        super().__init__()
        # Use better initialization for cross-dimensional mapping
        self.up = nn.Linear(dim_a, dim_b)
        self.down = nn.Linear(dim_b, dim_a)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.bias)
    
    def forward_up(self, x, use_dropout=True): 
        """Small model → Big model"""
        x = self.up(x)
        if use_dropout and self.training:
            x = self.dropout(x)
        return x
    
    def forward_down(self, x, use_dropout=True): 
        """Big model → Small model (for feature grafting)"""
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

print("Caching normalized activations for stitch training...")
train_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).take(ACTIVATION_CACHE_SIZE // 4)  # Reduce since we get ~128x more data per sample

acts_a_list, acts_b_list = [], []
processed_count = 0
total_positions_cached = 0

for item in tqdm(train_dataset, total=ACTIVATION_CACHE_SIZE // 4, desc="Caching activations"):
    try:
        tokens = torch.tensor(item['tokens'])[:CONTEXT_LENGTH]
        # Pad if necessary
        if len(tokens) < CONTEXT_LENGTH:
            tokens = torch.cat([tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)])
        tokens = tokens.to(device).unsqueeze(0)
        
        # Get activations from ALL sequence positions
        act_a = get_normalized_activations(model_a, tokens, STITCH_LAYER_A, return_all_positions=True)
        act_b = get_normalized_activations(model_b, tokens, STITCH_LAYER_B, return_all_positions=True)
        
        acts_a_list.append(act_a)
        acts_b_list.append(act_b)
        processed_count += 1
        total_positions_cached += act_a.shape[0]  # Should be CONTEXT_LENGTH (128)
        
        # Periodic memory cleanup
        if processed_count % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  Cached {total_positions_cached:,} total positions from {processed_count} sequences")
            
        # Stop if we have enough total positions (roughly equivalent to original cache size)
        if total_positions_cached >= ACTIVATION_CACHE_SIZE:
            print(f"Reached target of {ACTIVATION_CACHE_SIZE:,} positions, stopping early")
            break
            
    except Exception as e:
        print(f"Error caching activation: {e}")
        continue

print(f"Successfully cached {total_positions_cached:,} activation pairs from {len(acts_a_list)} sequences")
print(f"This is ~{total_positions_cached // (ACTIVATION_CACHE_SIZE // len(acts_a_list)) if len(acts_a_list) > 0 else 0}x more training data than the previous approach!")
torch.cuda.empty_cache()  # Clean up before stacking

# Concatenate all cached activations 
cached_acts_a_tensor = torch.cat(acts_a_list, dim=0).to(torch.bfloat16)  # [total_positions, dim_a]
cached_acts_b_tensor = torch.cat(acts_b_list, dim=0).to(torch.bfloat16)  # [total_positions, dim_b]
print(f"Final training data shape: A={cached_acts_a_tensor.shape}, B={cached_acts_b_tensor.shape}")

activation_dataset = ActivationDataset(cached_acts_a_tensor, cached_acts_b_tensor)
dataloader = DataLoader(activation_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize enhanced stitch
dim_a = model_a.config.hidden_size  # 2304 for Gemma 2B
dim_b = model_b.config.hidden_size  # 3584 for Gemma 9B
stitch_model = EnhancedStitch(dim_a, dim_b, dropout_rate=0.1).to(device)
optimizer = torch.optim.AdamW(stitch_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
loss_fn = nn.MSELoss()

print(f"\nEnhanced Stitch Architecture: {dim_a} -> {dim_b}")
print(f"Training data: {cached_acts_a_tensor.shape[0]:,} activation pairs from all sequence positions")
print(f"Training for {NUM_EPOCHS} epochs with {len(dataloader)} batches per epoch")
print("Starting stitch training...")

# Training loop with better monitoring
stitch_model.train()
loss_history = []
epoch_losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_up_loss = 0
    total_down_loss = 0
    total_cycle_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (h_a, h_b) in enumerate(progress_bar):
        h_a = h_a.to(device, dtype=torch.float32)
        h_b = h_b.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # Forward mappings
        h_a_to_b = stitch_model.forward_up(h_a)      # Small -> Big
        h_b_to_a = stitch_model.forward_down(h_b)    # Big -> Small (feature grafting direction!)
        
        # Cycle consistency
        h_a_cycle = stitch_model.forward_down(h_a_to_b, use_dropout=False)  # Small -> Big -> Small
        h_b_cycle = stitch_model.forward_up(h_b_to_a, use_dropout=False)    # Big -> Small -> Big
        
        # Component losses
        up_loss = loss_fn(h_a_to_b, h_b)      # Small model features -> Big model features
        down_loss = loss_fn(h_b_to_a, h_a)    # Big model features -> Small model features
        cycle_a_loss = loss_fn(h_a_cycle, h_a) # Cycle consistency for small model
        cycle_b_loss = loss_fn(h_b_cycle, h_b) # Cycle consistency for big model
        
        # Weighted combination (emphasize the direction you care about)
        # For feature grafting (big->small), you might want to weight down_loss more
        loss = up_loss + 1.2 * down_loss + 0.5 * (cycle_a_loss + cycle_b_loss)
        
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(stitch_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_up_loss += up_loss.item()
        total_down_loss += down_loss.item()
        total_cycle_loss += (cycle_a_loss.item() + cycle_b_loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Up': f"{up_loss.item():.4f}",
            'Down': f"{down_loss.item():.4f}",
            'LR': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        if batch_idx % 100 == 0:
            loss_history.append(loss.item())
    
    # End of epoch statistics
    avg_loss = total_loss / len(dataloader)
    avg_up_loss = total_up_loss / len(dataloader)
    avg_down_loss = total_down_loss / len(dataloader)
    avg_cycle_loss = total_cycle_loss / len(dataloader)
    
    epoch_losses.append(avg_loss)
    scheduler.step()
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Total Loss: {avg_loss:.4f}")
    print(f"  Small->Big Loss: {avg_up_loss:.4f}")
    print(f"  Big->Small Loss: {avg_down_loss:.4f} (Feature Grafting Direction)")
    print(f"  Cycle Loss: {avg_cycle_loss:.4f}")
    print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

print("Stitch training complete!")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, alpha=0.7, label='Batch Loss')
plt.title('Training Loss (Per Batch)')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epoch_losses, 'o-', linewidth=2, markersize=6)
plt.title('Average Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stitch_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Save the trained model
torch.save({
    'model_state_dict': stitch_model.state_dict(),
    'layer_a': STITCH_LAYER_A,
    'layer_b': STITCH_LAYER_B,
    'dim_a': dim_a,
    'dim_b': dim_b,
    'final_loss': avg_loss,
    'epoch_losses': epoch_losses
}, 'enhanced_stitch_model.pth')

print(f"Model saved as 'enhanced_stitch_model.pth'")

print("Loading Gemma Scope SAEs for evaluation...")

try:
    print(f"Loading SAE A: {SAE_A_RELEASE}/{SAE_A_ID}")
    sae_a = SAE.from_pretrained(release=SAE_A_RELEASE, sae_id=SAE_A_ID, device=device)
    
    print(f"Loading SAE B: {SAE_B_RELEASE}/{SAE_B_ID}")
    sae_b = SAE.from_pretrained(release=SAE_B_RELEASE, sae_id=SAE_B_ID, device=device)
    
    print("SAEs loaded successfully!")
    print(f"SAE A (Layer {STITCH_LAYER_A}): d_in={sae_a.cfg.d_in}, d_sae={sae_a.cfg.d_sae}")
    print(f"SAE B (Layer {STITCH_LAYER_B}): d_in={sae_b.cfg.d_in}, d_sae={sae_b.cfg.d_sae}")
    
except Exception as e:
    print(f"Warning: Could not load SAEs - {e}")
    print("Evaluation will proceed without SAE reconstruction")
    sae_a, sae_b = None, None

print("\n" + "="*60)
print("COMPREHENSIVE EVALUATION")
print("="*60)

stitch_model.eval()
if sae_a is not None:
    sae_a.eval()
if sae_b is not None:
    sae_b.eval()

# Get multiple test samples for robust evaluation
test_samples = []
test_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).skip(ACTIVATION_CACHE_SIZE // 4).take(10)
for item in test_dataset:
    test_tokens = torch.tensor(item['tokens'])[:CONTEXT_LENGTH]
    if len(test_tokens) < CONTEXT_LENGTH:
        test_tokens = torch.cat([test_tokens, torch.full((CONTEXT_LENGTH - len(test_tokens),), tokenizer.pad_token_id)])
    test_samples.append(test_tokens.to(device).unsqueeze(0))

all_transfer_losses = []
all_random_losses = []
all_reverse_losses = []  # Big -> Small direction

for i, test_tokens in enumerate(test_samples):
    print(f"\n--- Test Sample {i+1} ---")
    
    # Get activations from all positions
    try:
        h_A_unnormalized_all = get_unnormalized_activations(model_a, test_tokens, STITCH_LAYER_A, return_all_positions=True).to(device)
        h_B_unnormalized_all = get_unnormalized_activations(model_b, test_tokens, STITCH_LAYER_B, return_all_positions=True).to(device)
        
        h_A_normalized_all = get_normalized_activations(model_a, test_tokens, STITCH_LAYER_A, return_all_positions=True).to(device)
        h_B_normalized_all = get_normalized_activations(model_b, test_tokens, STITCH_LAYER_B, return_all_positions=True).to(device)
    except Exception as e:
        print(f"Error getting all positions, falling back to last token only: {e}")
        # Fallback to last token only
        h_A_unnormalized_all = get_unnormalized_activations(model_a, test_tokens, STITCH_LAYER_A, return_all_positions=False).to(device).unsqueeze(0)
        h_B_unnormalized_all = get_unnormalized_activations(model_b, test_tokens, STITCH_LAYER_B, return_all_positions=False).to(device).unsqueeze(0)
        
        h_A_normalized_all = get_normalized_activations(model_a, test_tokens, STITCH_LAYER_A, return_all_positions=False).to(device).unsqueeze(0)
        h_B_normalized_all = get_normalized_activations(model_b, test_tokens, STITCH_LAYER_B, return_all_positions=False).to(device).unsqueeze(0)
    
    # For evaluation, use a few representative positions (start, middle, end)
    seq_len = h_A_normalized_all.shape[0]
    if seq_len > 1:
        eval_positions = [0, seq_len // 2, seq_len - 1]  # Start, middle, end
    else:
        eval_positions = [0]  # Just one position if fallback was used
    
    position_transfer_losses = []
    position_reverse_losses = []
    position_random_losses = []
    
    for pos_idx in eval_positions:
        h_A_unnormalized = h_A_unnormalized_all[pos_idx]
        h_B_unnormalized = h_B_unnormalized_all[pos_idx] 
        h_A_normalized = h_A_normalized_all[pos_idx]
        h_B_normalized = h_B_normalized_all[pos_idx]
        
        # SAE processing (if available)
        if sae_a is not None:
            try:
                sae_a_output = sae_a(h_A_unnormalized.unsqueeze(0))
                h_A_reconstructed_unnormalized = sae_a_output[0].squeeze(0)
                h_A_reconstructed_normalized = torch.nn.functional.layer_norm(
                    h_A_reconstructed_unnormalized, 
                    [h_A_reconstructed_unnormalized.shape[-1]]
                )
            except:
                h_A_reconstructed_normalized = h_A_normalized
        else:
            h_A_reconstructed_normalized = h_A_normalized
        
        with torch.no_grad():
            # Forward direction: Small -> Big
            h_A_transferred_normalized = stitch_model.forward_up(h_A_reconstructed_normalized, use_dropout=False)
            transfer_loss = loss_fn(h_A_transferred_normalized, h_B_normalized).item()
            
            # Reverse direction: Big -> Small (Feature Grafting!)
            h_B_transferred_normalized = stitch_model.forward_down(h_B_normalized, use_dropout=False)
            reverse_loss = loss_fn(h_B_transferred_normalized, h_A_normalized).item()
            
            # Random baselines
            random_up = nn.Linear(dim_a, dim_b).to(device)
            h_A_random = random_up(h_A_reconstructed_normalized)
            
            random_up_loss = loss_fn(h_A_random, h_B_normalized).item()
        
        position_transfer_losses.append(transfer_loss)
        position_reverse_losses.append(reverse_loss)
        position_random_losses.append(random_up_loss)
    
    # Average across positions
    avg_transfer = np.mean(position_transfer_losses)
    avg_reverse = np.mean(position_reverse_losses)
    avg_random = np.mean(position_random_losses)
    
    all_transfer_losses.append(avg_transfer)
    all_random_losses.append(avg_random)
    all_reverse_losses.append(avg_reverse)
    
    print(f"  Small->Big Transfer MSE (avg): {avg_transfer:.4f}")
    print(f"  Big->Small Transfer MSE (avg): {avg_reverse:.4f} ← Feature Grafting Direction")
    print(f"  Random Baseline MSE (avg): {avg_random:.4f}")
    print(f"  Positions evaluated: {eval_positions} (start, middle, end)")

# Final summary
avg_transfer = np.mean(all_transfer_losses)
avg_random = np.mean(all_random_losses)
avg_reverse = np.mean(all_reverse_losses)

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Average Small->Big Transfer MSE:     {avg_transfer:.4f}")
print(f"Average Big->Small Transfer MSE:     {avg_reverse:.4f} ← FEATURE GRAFTING")
print(f"Average Random Baseline MSE:        {avg_random:.4f}")
print("-" * 60)

# Success metrics
up_improvement = avg_random / avg_transfer if avg_transfer > 0 else float('inf')
down_improvement = avg_random / avg_reverse if avg_reverse > 0 else float('inf')

print(f"Small->Big Improvement Factor:       {up_improvement:.2f}x")
print(f"Big->Small Improvement Factor:       {down_improvement:.2f}x ← KEY METRIC")

success_threshold = 2.0  # At least 2x better than random
up_success = up_improvement > success_threshold
down_success = down_improvement > success_threshold

print("\n" + "="*60)
if up_success and down_success:
    print("SUCCESS: Both directions show significant improvement!")
    print(f"   Ready for feature grafting experiments (Big->Small: {down_improvement:.1f}x better)")
elif down_success:
    print("PARTIAL SUCCESS: Feature grafting direction works well!")
    print(f"   Big->Small transfer is {down_improvement:.1f}x better than random")
elif up_success:
    print("PARTIAL SUCCESS: Small->Big direction works well!")
    print(f"   Small->Big transfer is {up_improvement:.1f}x better than random")
else:
    print("NEEDS IMPROVEMENT: Consider training longer or adjusting hyperparameters")

print(f"\nLayer Pair Used: Gemma 2B Layer {STITCH_LAYER_A} → Gemma 9B Layer {STITCH_LAYER_B}")
print(f"Training completed with {NUM_EPOCHS} epochs")
print("="*60)
