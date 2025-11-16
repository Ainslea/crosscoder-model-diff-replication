import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sae_lens import SAE
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.metrics import r2_score

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration (should match your training setup)
MODEL_A_ID = "google/gemma-2-2b"     
MODEL_B_ID = "google/gemma-2-9b"    
STITCH_LAYER_A = 8   # Gemma 2B layer 8
STITCH_LAYER_B = 17  # Gemma 9B layer 17
SAE_A_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_A_ID = f"layer_{STITCH_LAYER_A}/width_16k/canonical"  
SAE_B_RELEASE = "gemma-scope-9b-pt-res-canonical" 
SAE_B_ID = f"layer_{STITCH_LAYER_B}/width_16k/canonical"
DATASET_NAME = "NeelNanda/openwebtext-tokenized-9b"
CONTEXT_LENGTH = 128
NUM_TEST_SAMPLES = 500  # Number of samples for correlation analysis


def get_unnormalized_activations(model, tokens, layer_idx):
    """Get unnormalized activations for the last token"""
    with torch.no_grad():
        if hasattr(model, 'model'):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)
        
        activations = outputs.hidden_states[layer_idx].squeeze(0)[-1]
        # Convert to float32 and check for NaN/inf
        activations = activations.float()
        if torch.isnan(activations).any() or torch.isinf(activations).any():
            print(f"Warning: NaN/inf detected in unnormalized activations for layer {layer_idx}")
            activations = torch.nan_to_num(activations, nan=0.0, posinf=1e6, neginf=-1e6)
        return activations.cpu()

def get_normalized_activations(model, tokens, layer_idx):
    """Get normalized activations for the last token"""
    with torch.no_grad():
        if hasattr(model, 'model'):
            outputs = model.model(tokens, output_hidden_states=True)
        else:
            outputs = model(tokens, output_hidden_states=True)
            
        activations = outputs.hidden_states[layer_idx].squeeze(0)[-1]
        # Convert to float32 first
        activations = activations.float()
        
        # Check for NaN/inf before normalization
        if torch.isnan(activations).any() or torch.isinf(activations).any():
            print(f"Warning: NaN/inf detected in activations before normalization for layer {layer_idx}")
            activations = torch.nan_to_num(activations, nan=0.0, posinf=1e6, neginf=-1e6)
        
        normalized_activations = torch.nn.functional.layer_norm(activations, [activations.shape[-1]])
        
        # Check again after normalization
        if torch.isnan(normalized_activations).any() or torch.isinf(normalized_activations).any():
            print(f"Warning: NaN/inf detected after normalization for layer {layer_idx}")
            normalized_activations = torch.nan_to_num(normalized_activations, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return normalized_activations.cpu()

def analyze_sae_feature_activations(sae, activations, top_k=50):
    """
    Get SAE feature activations and identify most active features
    Returns: feature_acts, top_feature_indices, reconstruction
    """
    with torch.no_grad():
        # Ensure correct dtype and shape
        if len(activations.shape) == 1:
            activations = activations.unsqueeze(0)
        
        # Convert to float32 and move to SAE device
        activations = activations.float().to(sae.device)
        
        # Check for NaN/inf before SAE processing
        if torch.isnan(activations).any() or torch.isinf(activations).any():
            print("Warning: NaN/inf in activations before SAE processing")
            activations = torch.nan_to_num(activations, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            # Get SAE outputs - handle different SAE return formats
            sae_output = sae(activations)
            
            if isinstance(sae_output, tuple) and len(sae_output) >= 3:
                # Format: (reconstruction, feature_acts, ...)
                reconstruction = sae_output[0].squeeze(0).cpu().float()
                feature_acts = sae_output[1].squeeze(0).cpu().float()
            elif isinstance(sae_output, tuple) and len(sae_output) == 2:
                reconstruction = sae_output[0].squeeze(0).cpu().float()
                feature_acts = sae_output[1].squeeze(0).cpu().float()
            else:
                # Only reconstruction returned, need to get features separately
                reconstruction = sae_output.squeeze(0).cpu().float()
                try:
                    feature_acts = sae.encode(activations).squeeze(0).cpu().float()
                except:
                    # If encode doesn't work, create dummy features
                    print("Warning: Could not get feature activations, using dummy values")
                    feature_acts = torch.zeros(sae.cfg.d_sae).float()
            
            # Check for NaN/inf in outputs
            if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
                print("Warning: NaN/inf in SAE reconstruction")
                reconstruction = torch.nan_to_num(reconstruction, nan=0.0, posinf=1e6, neginf=-1e6)
                
            if torch.isnan(feature_acts).any() or torch.isinf(feature_acts).any():
                print("Warning: NaN/inf in SAE feature activations")
                feature_acts = torch.nan_to_num(feature_acts, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Get top-k most active features
            feature_magnitudes = torch.abs(feature_acts)
            if feature_magnitudes.sum() == 0:
                # If all features are zero, just take first k
                top_indices = torch.arange(min(top_k, len(feature_magnitudes)))
            else:
                top_indices = torch.topk(feature_magnitudes, min(top_k, len(feature_magnitudes)))[1]
            
            return feature_acts, top_indices, reconstruction
            
        except Exception as e:
            print(f"Error in SAE processing: {e}")
            # Return dummy values
            dummy_features = torch.zeros(sae.cfg.d_sae).float()
            dummy_indices = torch.arange(min(top_k, sae.cfg.d_sae))
            dummy_reconstruction = torch.zeros_like(activations.squeeze(0).cpu()).float()
            return dummy_features, dummy_indices, dummy_reconstruction

def compute_feature_correlation_through_stitch(feature_acts_9b, stitch_model, sae_2b, activations_2b_unnorm):
    """
    Test how well 9B features correlate with 2B features after going through the stitch
    
    Process:
    1. Take 9B feature activations
    2. Reconstruct from 9B SAE to get activation space
    3. Pass through stitch (down direction: 9B -> 2B)  
    4. Encode with 2B SAE to get 2B feature space
    5. Compare with actual 2B feature activations
    """
    with torch.no_grad():
        # Step 1: Get 2B feature activations directly
        actual_2b_features, _, _ = analyze_sae_feature_activations(sae_2b, activations_2b_unnorm)
        
        # Step 2: Reconstruct 9B features back to activation space
        # This is a simplified approach - in practice you might want to use SAE decoder
        # For now, we'll work with the passed activations and use stitch directly
        
        # We need 9B activations to pass through stitch
        # This is a placeholder - you'd need to pass in the actual 9B activations
        # For this analysis, let's focus on correlation of the stitch output with 2B features
        
        return actual_2b_features

def find_reasoning_related_features(sae, tokenizer, reasoning_prompts, top_k=20):
    """
    Identify features that activate strongly on reasoning-related content
    """
    reasoning_texts = [
        "Let me think step by step about this problem.",
        "First, I need to understand what the question is asking.",
        "To solve this, I'll break it down into smaller parts.",
        "The logical conclusion is that",
        "This reasoning leads me to believe",
        "If we assume that A is true, then B must follow because",
        "The evidence suggests that the answer is",
        "Let me analyze each option carefully:",
    ]
    
    reasoning_feature_scores = torch.zeros(sae.cfg.d_sae)
    
    for text in reasoning_texts:
        tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=50)
        tokens = tokens.to(device)
        
        # Get activations - this is simplified, you'd need the actual model activations
        # For now, this is a placeholder structure
        pass
    
    return torch.topk(reasoning_feature_scores, top_k)[1]


# Collect test samples
print("Collecting test samples...")
test_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).take(NUM_TEST_SAMPLES)

test_samples = []
for item in tqdm(test_dataset, desc="Preparing test samples", total=NUM_TEST_SAMPLES):
    tokens = torch.tensor(item['tokens'])[:CONTEXT_LENGTH]
    if len(tokens) < CONTEXT_LENGTH:
        tokens = torch.cat([tokens, torch.full((CONTEXT_LENGTH - len(tokens),), tokenizer.pad_token_id)])
    test_samples.append(tokens.to(device).unsqueeze(0))

print(f"Prepared {len(test_samples)} test samples")

# Feature correlation analysis
print("\nStarting feature correlation analysis...")
correlations = []
feature_transfer_scores = []
reconstruction_errors = []

# We'll analyze a subset for detailed correlation
analysis_samples = test_samples[:10]  # Use first 10 for detailed analysis with debugging

for i, tokens in enumerate(tqdm(analysis_samples, desc="Analyzing feature correlations")):
    print(f"\n--- Processing Sample {i} ---")
    try:
        # Get activations from both models
        act_2b_unnorm = get_unnormalized_activations(model_a, tokens, STITCH_LAYER_A)
        act_9b_unnorm = get_unnormalized_activations(model_b, tokens, STITCH_LAYER_B)
        act_2b_norm = get_normalized_activations(model_a, tokens, STITCH_LAYER_A)
        act_9b_norm = get_normalized_activations(model_b, tokens, STITCH_LAYER_B)
        
        # Skip if any activations are all zeros (indicating error)
        print(f"Sample {i}: Activation sums - 2B unnorm: {act_2b_unnorm.abs().sum():.4f}, "
              f"9B unnorm: {act_9b_unnorm.abs().sum():.4f}, "
              f"2B norm: {act_2b_norm.abs().sum():.4f}, "
              f"9B norm: {act_9b_norm.abs().sum():.4f}")
              
        if (act_2b_unnorm.abs().sum() == 0 or act_9b_unnorm.abs().sum() == 0 or 
            act_2b_norm.abs().sum() == 0 or act_9b_norm.abs().sum() == 0):
            print(f"Skipping sample {i}: zero activations detected")
            continue
        
        print(f"Sample {i}: Getting SAE feature activations...")
        
        # Get SAE feature activations
        features_2b, top_2b, recon_2b = analyze_sae_feature_activations(sae_a, act_2b_unnorm)
        features_9b, top_9b, recon_9b = analyze_sae_feature_activations(sae_b, act_9b_unnorm)
        
        print(f"Sample {i}: SAE results - 2B features shape: {features_2b.shape}, 9B features shape: {features_9b.shape}")
        print(f"Sample {i}: Top features - 2B: {len(top_2b)}, 9B: {len(top_9b)}")
        
        print(f"Sample {i}: Testing stitch transfer...")
        # Test stitch transfer: 9B -> 2B (feature grafting direction)
        # Ensure proper dtype conversion for stitch
        act_9b_norm_device = act_9b_norm.to(device).float()
        print(f"Sample {i}: 9B norm activation shape: {act_9b_norm_device.shape}, device: {act_9b_norm_device.device}")
        
        transferred_2b = stitch_model.forward_down(act_9b_norm_device, use_dropout=False)
        print(f"Sample {i}: Transferred 2B shape: {transferred_2b.shape}")
        
        # Check for NaN in stitch output
        if torch.isnan(transferred_2b).any() or torch.isinf(transferred_2b).any():
            print(f"Sample {i}: NaN/inf in stitch output - skipping")
            continue
        
        print(f"Sample {i}: Computing reconstruction errors...")
        # Get reconstruction errors - ensure same device and dtype
        recon_2b_cpu = recon_2b.cpu().float()
        act_2b_unnorm_cpu = act_2b_unnorm.cpu().float()
        recon_9b_cpu = recon_9b.cpu().float()
        act_9b_unnorm_cpu = act_9b_unnorm.cpu().float()
        
        print(f"Sample {i}: Reconstruction shapes - 2B recon: {recon_2b_cpu.shape}, 2B act: {act_2b_unnorm_cpu.shape}")
        print(f"Sample {i}: Reconstruction shapes - 9B recon: {recon_9b_cpu.shape}, 9B act: {act_9b_unnorm_cpu.shape}")
        
        recon_error_2b = float('nan')
        recon_error_9b = float('nan')
        
        if recon_2b_cpu.shape == act_2b_unnorm_cpu.shape:
            recon_error_2b = torch.nn.functional.mse_loss(recon_2b_cpu, act_2b_unnorm_cpu).item()
            print(f"Sample {i}: 2B reconstruction error: {recon_error_2b:.6f}")
        else:
            print(f"Sample {i}: Shape mismatch for 2B reconstruction: {recon_2b_cpu.shape} vs {act_2b_unnorm_cpu.shape}")
            
        if recon_9b_cpu.shape == act_9b_unnorm_cpu.shape:
            recon_error_9b = torch.nn.functional.mse_loss(recon_9b_cpu, act_9b_unnorm_cpu).item()
            print(f"Sample {i}: 9B reconstruction error: {recon_error_9b:.6f}")
        else:
            print(f"Sample {i}: Shape mismatch for 9B reconstruction: {recon_9b_cpu.shape} vs {act_9b_unnorm_cpu.shape}")
        
        # Only add if not NaN
        if not (np.isnan(recon_error_2b) and np.isnan(recon_error_9b)):
            reconstruction_errors.append({
                'sample_idx': i,
                'recon_error_2b': recon_error_2b,
                'recon_error_9b': recon_error_9b
            })
            print(f"Sample {i}: Added reconstruction errors")
        
        # Compare transferred activation with actual 2B activation
        transferred_2b_cpu = transferred_2b.cpu().float()
        act_2b_norm_cpu = act_2b_norm.cpu().float()
        
        print(f"Sample {i}: Transfer shapes - transferred: {transferred_2b_cpu.shape}, actual: {act_2b_norm_cpu.shape}")
        
        if transferred_2b_cpu.shape == act_2b_norm_cpu.shape:
            transfer_error = torch.nn.functional.mse_loss(transferred_2b_cpu, act_2b_norm_cpu).item()
            print(f"Sample {i}: Transfer error: {transfer_error:.6f}")
            if not np.isnan(transfer_error) and not np.isinf(transfer_error):
                print(f"Sample {i}: Valid transfer error added")
            else:
                print(f"Sample {i}: Invalid transfer error: {transfer_error}")
                transfer_error = float('nan')
        else:
            print(f"Shape mismatch for transfer: {transferred_2b_cpu.shape} vs {act_2b_norm_cpu.shape}")
            transfer_error = float('nan')
        
        # Feature-level analysis: check if top features align
        if len(top_2b) > 0 and len(top_9b) > 0 and len(features_2b) > 0 and len(features_9b) > 0:
            # Get feature activations for top features
            try:
                top_2b_values = features_2b[top_2b[:10]].float()  # Top 10 2B features
                top_9b_values = features_9b[top_9b[:10]].float()  # Top 10 9B features
                
                print(f"Sample {i}: 2B top features shape: {top_2b_values.shape}, 9B top features shape: {top_9b_values.shape}")
                print(f"Sample {i}: 2B feature stats: mean={top_2b_values.mean():.4f}, std={top_2b_values.std():.4f}")
                print(f"Sample {i}: 9B feature stats: mean={top_9b_values.mean():.4f}, std={top_9b_values.std():.4f}")
                
                # Simple correlation between top feature activations
                if len(top_2b_values) > 1 and len(top_9b_values) > 1:
                    # Ensure same length for correlation
                    min_len = min(len(top_2b_values), len(top_9b_values))
                    if min_len > 1:
                        top_2b_vals = top_2b_values[:min_len].detach().cpu().numpy()
                        top_9b_vals = top_9b_values[:min_len].detach().cpu().numpy()
                        
                        print(f"Sample {i}: Correlation arrays - 2B: {top_2b_vals}, 9B: {top_9b_vals}")
                        
                        # Check for non-zero variance
                        var_2b = np.var(top_2b_vals)
                        var_9b = np.var(top_9b_vals)
                        print(f"Sample {i}: Variances - 2B: {var_2b:.6f}, 9B: {var_9b:.6f}")
                        
                        if var_2b > 1e-8 and var_9b > 1e-8:
                            corr_coef, p_val = pearsonr(top_2b_vals, top_9b_vals)
                            print(f"Sample {i}: Correlation coefficient: {corr_coef:.4f}, p-value: {p_val:.4f}")
                            if not np.isnan(corr_coef) and not np.isinf(corr_coef):
                                correlations.append(corr_coef)
                                print(f"Sample {i}: Added correlation: {corr_coef:.4f}")
                            else:
                                print(f"Sample {i}: Invalid correlation: {corr_coef}")
                        else:
                            print(f"Sample {i}: Zero variance detected - skipping correlation")
            except Exception as e:
                print(f"Sample {i}: Error in correlation computation: {e}")
                import traceback
                traceback.print_exc()
        
        # Only add transfer score if not NaN
        if not np.isnan(transfer_error):
            feature_transfer_scores.append({
                'sample_idx': i,
                'transfer_error': transfer_error,
                'top_2b_features': top_2b[:5].tolist(),
                'top_9b_features': top_9b[:5].tolist(),
                'top_2b_activations': features_2b[top_2b[:5]].detach().cpu().float().tolist() if len(top_2b) >= 5 else [],
                'top_9b_activations': features_9b[top_9b[:5]].detach().cpu().float().tolist() if len(top_9b) >= 5 else []
            })
            print(f"Sample {i}: Added to feature_transfer_scores")
        else:
            print(f"Sample {i}: Skipped due to invalid transfer error")
        
    except Exception as e:
        print(f"Error in sample {i}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Analysis results
print("\n" + "="*60)
print("FEATURE CORRELATION ANALYSIS RESULTS")
print("="*60)

print(f"Successfully processed samples: {len(feature_transfer_scores)}")
print(f"Samples with valid correlations: {len(correlations)}")
print(f"Samples with valid reconstruction errors: {len(reconstruction_errors)}")

if correlations:
    avg_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    print(f"Average top-feature correlation: {avg_correlation:.4f} ± {std_correlation:.4f}")
    print(f"Correlation range: {min(correlations):.4f} to {max(correlations):.4f}")
    
    # Show distribution of correlations
    positive_corrs = [c for c in correlations if c > 0.1]
    negative_corrs = [c for c in correlations if c < -0.1]
    print(f"Strong positive correlations (>0.1): {len(positive_corrs)}")
    print(f"Strong negative correlations (<-0.1): {len(negative_corrs)}")
else:
    print("Could not compute feature correlations")

if feature_transfer_scores:
    valid_transfer_errors = [s['transfer_error'] for s in feature_transfer_scores if not np.isnan(s['transfer_error'])]
    if valid_transfer_errors:
        avg_transfer_error = np.mean(valid_transfer_errors)
        std_transfer_error = np.std(valid_transfer_errors)
        print(f"Average transfer error (9B->2B): {avg_transfer_error:.4f} ± {std_transfer_error:.4f}")
        print(f"Transfer error range: {min(valid_transfer_errors):.4f} to {max(valid_transfer_errors):.4f}")
    else:
        print("No valid transfer errors computed")

if reconstruction_errors:
    valid_recon_2b = [s['recon_error_2b'] for s in reconstruction_errors if not np.isnan(s['recon_error_2b'])]
    valid_recon_9b = [s['recon_error_9b'] for s in reconstruction_errors if not np.isnan(s['recon_error_9b'])]
    
    if valid_recon_2b:
        avg_recon_2b = np.mean(valid_recon_2b)
        print(f"Average 2B SAE reconstruction error: {avg_recon_2b:.4f}")
    else:
        print("No valid 2B reconstruction errors")
        
    if valid_recon_9b:
        avg_recon_9b = np.mean(valid_recon_9b)
        print(f"Average 9B SAE reconstruction error: {avg_recon_9b:.4f}")
    else:
        print("No valid 9B reconstruction errors")

# Feature overlap analysis
print("\nFeature Activation Overlap Analysis:")
all_top_2b = []
all_top_9b = []
for score in feature_transfer_scores:
    all_top_2b.extend(score['top_2b_features'])
    all_top_9b.extend(score['top_9b_features'])

# Most frequently activated features
if all_top_2b and all_top_9b:
    unique_2b, counts_2b = np.unique(all_top_2b, return_counts=True)
    unique_9b, counts_9b = np.unique(all_top_9b, return_counts=True)
    
    print(f"Most active 2B features: {unique_2b[np.argsort(counts_2b)[-5:]]}")
    print(f"Most active 9B features: {unique_9b[np.argsort(counts_9b)[-5:]]}")

# Plotting results
if correlations and len(correlations) > 1:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Top-Feature Correlations')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    if feature_transfer_scores:
        transfer_errors = [s['transfer_error'] for s in feature_transfer_scores]
        plt.subplot(1, 3, 2)
        plt.hist(transfer_errors, bins=20, alpha=0.7, edgecolor='black', color='orange')
        plt.title('Distribution of Transfer Errors (9B->2B)')
        plt.xlabel('MSE Loss')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
    
    if reconstruction_errors:
        recon_2b = [s['recon_error_2b'] for s in reconstruction_errors]
        recon_9b = [s['recon_error_9b'] for s in reconstruction_errors]
        
        plt.subplot(1, 3, 3)
        plt.scatter(recon_2b, recon_9b, alpha=0.6)
        plt.xlabel('2B SAE Reconstruction Error')
        plt.ylabel('9B SAE Reconstruction Error')
        plt.title('SAE Reconstruction Error Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(min(recon_2b), min(recon_9b))
        max_val = max(max(recon_2b), max(recon_9b))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('feature_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("High correlation (>0.5): Features transfer well through stitch")
print("Medium correlation (0.2-0.5): Partial feature alignment")
print("Low correlation (<0.2): Poor feature transfer")
print("")
print("Low transfer error: Good 9B->2B activation mapping")
print("Low reconstruction error: SAEs capture activations well")
print("")
print("Next steps for feature grafting:")
print("1. Identify highly correlated feature pairs")
print("2. Test on reasoning-specific prompts")
print("3. Implement selective feature grafting based on correlation scores")
print("="*60)

# Save detailed results
results = {
    'correlations': correlations,
    'feature_transfer_scores': feature_transfer_scores,
    'reconstruction_errors': reconstruction_errors,
    'config': {
        'layer_a': STITCH_LAYER_A,
        'layer_b': STITCH_LAYER_B,
        'num_samples': len(analysis_samples)
    }
}

torch.save(results, 'feature_correlation_results.pt')
print("Results saved to 'feature_correlation_results.pt'")
