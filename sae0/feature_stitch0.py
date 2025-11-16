import os
import torch

POD_VOLUME_PATH = "/workspace" 
# Alternative common paths: "/content", "/data", "/volume", etc.

# Create cache directory
CACHE_DIR = os.path.join(POD_VOLUME_PATH, "stitch_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def save_activations_cache(acts_a, acts_b, cache_path):
    """Save activation cache to disk"""
    print(f"Saving activation cache to {cache_path}...")
    cache_data = {
        'acts_a': acts_a,
        'acts_b': acts_b,
        'layer_a': STITCH_LAYER_A,
        'layer_b': STITCH_LAYER_B,
        'context_length': CONTEXT_LENGTH,
        'timestamp': torch.tensor([0])  # Placeholder for metadata
    }
    torch.save(cache_data, cache_path)
    print(f"✓ Saved {acts_a.shape[0]:,} activation pairs")

def load_activations_cache(cache_path):
    """Load activation cache from disk"""
    if os.path.exists(cache_path):
        print(f"Loading activation cache from {cache_path}...")
        cache_data = torch.load(cache_path)
        print(f"✓ Loaded {cache_data['acts_a'].shape[0]:,} activation pairs")
        print(f"  Cached for layers: {cache_data['layer_a']} -> {cache_data['layer_b']}")
        return cache_data['acts_a'], cache_data['acts_b']
    return None, None

# Define cache file paths
activation_cache_file = os.path.join(CACHE_DIR, f"activations_L{STITCH_LAYER_A}to{STITCH_LAYER_B}_ctx{CONTEXT_LENGTH}.pt")
model_save_path = os.path.join(CACHE_DIR, f"stitch_model_L{STITCH_LAYER_A}to{STITCH_LAYER_B}.pt")

print(f"Cache directory: {CACHE_DIR}")
print(f"Activation cache: {activation_cache_file}")
print(f"Model save path: {model_save_path}")

# Try to load existing activation cache
print("Checking for existing activation cache...")
cached_acts_a_tensor, cached_acts_b_tensor = load_activations_cache(activation_cache_file)

if cached_acts_a_tensor is not None and cached_acts_b_tensor is not None:
    print("Using cached activations! Skipping activation extraction.")
    
    # Verify cache is compatible
    expected_dim_a = model_a.config.hidden_size
    expected_dim_b = model_b.config.hidden_size
    
    if (cached_acts_a_tensor.shape[1] == expected_dim_a and 
        cached_acts_b_tensor.shape[1] == expected_dim_b):
        print("Cache dimensions match current models")
    else:
        print("Cache dimensions don't match - will regenerate")
        cached_acts_a_tensor, cached_acts_b_tensor = None, None
        
else:
    print("No compatible cache found - will generate activations")

# Generate activations if no cache or cache invalid
if cached_acts_a_tensor is None or cached_acts_b_tensor is None:
    print("Caching normalized activations for stitch training...")
    print("Now using ALL sequence positions for much richer training data!")
    train_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).take(ACTIVATION_CACHE_SIZE // 4)

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
            total_positions_cached += act_a.shape[0]
            
            # Periodic memory cleanup and progress updates
            if processed_count % 100 == 0:
                torch.cuda.empty_cache()
                print(f"  Cached {total_positions_cached:,} total positions from {processed_count} sequences")
                
                # Save intermediate cache every 500 samples (in case of interruption)
                if processed_count % 500 == 0:
                    temp_acts_a = torch.cat(acts_a_list, dim=0).to(torch.bfloat16)
                    temp_acts_b = torch.cat(acts_b_list, dim=0).to(torch.bfloat16)
                    temp_cache_path = activation_cache_file.replace('.pt', f'_temp_{processed_count}.pt')
                    save_activations_cache(temp_acts_a, temp_acts_b, temp_cache_path)
                
            # Stop if we have enough total positions
            if total_positions_cached >= ACTIVATION_CACHE_SIZE:
                print(f"Reached target of {ACTIVATION_CACHE_SIZE:,} positions, stopping early")
                break
                
        except Exception as e:
            print(f"Error caching activation: {e}")
            continue

    print(f"Successfully cached {total_positions_cached:,} activation pairs from {len(acts_a_list)} sequences")
    torch.cuda.empty_cache()

    # Concatenate and save to cache
    cached_acts_a_tensor = torch.cat(acts_a_list, dim=0).to(torch.bfloat16)
    cached_acts_b_tensor = torch.cat(acts_b_list, dim=0).to(torch.bfloat16)
    
    # Save the final cache
    save_activations_cache(cached_acts_a_tensor, cached_acts_b_tensor, activation_cache_file)
    
    # Clean up temporary files
    for temp_file in os.listdir(CACHE_DIR):
        if temp_file.startswith(f"activations_L{STITCH_LAYER_A}to{STITCH_LAYER_B}_ctx{CONTEXT_LENGTH}_temp_"):
            temp_path = os.path.join(CACHE_DIR, temp_file)
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_file}")

print(f"Final training data shape: A={cached_acts_a_tensor.shape}, B={cached_acts_b_tensor.shape}")

# Continue with training setup
activation_dataset = ActivationDataset(cached_acts_a_tensor, cached_acts_b_tensor)
dataloader = DataLoader(activation_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Enhanced model saving with metadata
def save_trained_model(model, save_path, training_info):
    """Save model with comprehensive metadata"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer_a': STITCH_LAYER_A,
        'layer_b': STITCH_LAYER_B,
        'dim_a': training_info['dim_a'],
        'dim_b': training_info['dim_b'],
        'epochs_trained': training_info['epochs'],
        'final_loss': training_info['final_loss'],
        'epoch_losses': training_info['epoch_losses'],
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'context_length': CONTEXT_LENGTH,
        'activation_cache_size': cached_acts_a_tensor.shape[0],
        'model_a_id': MODEL_A_ID,
        'model_b_id': MODEL_B_ID,
        'timestamp': torch.tensor([0])  # Could add actual timestamp
    }, save_path)
    print(f"✓ Model saved to {save_path}")

# Check for existing trained model
def load_trained_model(model, load_path):
    """Load a previously trained model"""
    if os.path.exists(load_path):
        print(f"Found existing model at {load_path}")
        checkpoint = torch.load(load_path)
        
        # Verify compatibility
        if (checkpoint['layer_a'] == STITCH_LAYER_A and 
            checkpoint['layer_b'] == STITCH_LAYER_B and
            checkpoint['dim_a'] == model.up.in_features and
            checkpoint['dim_b'] == model.up.out_features):
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded compatible trained model!")
            print(f"  Trained for {checkpoint['epochs_trained']} epochs")
            print(f"  Final loss: {checkpoint['final_loss']:.4f}")
            return True, checkpoint
        else:
            print("Model architecture mismatch - will train from scratch")
    
    return False, None


print(f"\nCache Management Summary:")
print(f"   Activation cache: {os.path.getsize(activation_cache_file) / (1024**3):.2f} GB" if os.path.exists(activation_cache_file) else "   No activation cache yet")
print(f"   Model save path ready: {model_save_path}")
print(f"   Total cache directory: {sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)) / (1024**3):.2f} GB" if os.path.exists(CACHE_DIR) else "   Cache directory created")
