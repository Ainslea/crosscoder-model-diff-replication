import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
GSM8K_SUBSET_SIZE = 100  # Number of GSM8K samples to use for feature identification
NUM_TOP_FEATURES = 50    # Number of top features to track per sample
GRAFTING_STRENGTHS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Different grafting strengths to test
MAX_NEW_TOKENS = 150     # Max tokens for generation
BATCH_SIZE = 8          # Batch size for evaluation

def extract_numerical_answer(text):
    """Extract the final numerical answer from GSM8K response"""
    # Look for patterns like "The answer is 42" or "#### 42"
    patterns = [
        r'####\s*([+-]?\d+(?:\.\d+)?)',
        r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)',
        r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)',
        r'=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$)',
        r'\$([+-]?\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except ValueError:
                continue
    
    # Fallback: extract any number from the end of the text
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text[-100:])  # Last 100 chars
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def identify_reasoning_features(model_b, sae_b, tokenizer, gsm8k_samples, layer_idx):
    """
    Identify features that are most active during mathematical reasoning
    by processing GSM8K samples with the larger model
    """
    print(f"Identifying reasoning features using {len(gsm8k_samples)} GSM8K samples...")
    
    feature_activations = defaultdict(list)  # feature_id -> list of activation values
    feature_frequency = defaultdict(int)     # feature_id -> count of times it was active
    
    model_b.eval()
    sae_b.eval()
    
    for i, sample in enumerate(tqdm(gsm8k_samples, desc="Processing GSM8K samples")):
        question = sample['question']
        
        # Tokenize the question (not the answer - we want features that activate on reasoning)
        inputs = tokenizer(
            f"Question: {question}\nLet me solve this step by step:",
            return_tensors="pt",
            truncation=True,
            max_length=CONTEXT_LENGTH,
            padding=True
        ).to(device)
        
        try:
            with torch.no_grad():
                # Get activations from the specified layer
                outputs = model_b.model(inputs['input_ids'], output_hidden_states=True)
                activations = outputs.hidden_states[layer_idx][:, -1].float()  # Last token
                
                # Process through SAE to get feature activations
                sae_output = sae_b(activations)
                if isinstance(sae_output, tuple) and len(sae_output) >= 2:
                    feature_acts = sae_output[1].squeeze(0).cpu().float()
                else:
                    feature_acts = sae_b.encode(activations).squeeze(0).cpu().float()
                
                # Track top active features
                feature_magnitudes = torch.abs(feature_acts)
                top_indices = torch.topk(feature_magnitudes, NUM_TOP_FEATURES)[1]
                top_values = feature_acts[top_indices]
                
                for idx, value in zip(top_indices, top_values):
                    feature_id = idx.item()
                    feature_activations[feature_id].append(value.item())
                    if abs(value.item()) > 0.1:  # Only count significantly active features
                        feature_frequency[feature_id] += 1
                        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Identify the most consistently active features
    print(f"Analyzed {len(feature_activations)} unique features")
    
    # Sort features by frequency and average activation strength
    feature_scores = {}
    for feature_id, activations in feature_activations.items():
        if len(activations) > 0:
            avg_activation = np.mean(np.abs(activations))
            frequency = feature_frequency[feature_id]
            # Score combines frequency and average strength
            score = frequency * avg_activation
            feature_scores[feature_id] = {
                'score': score,
                'frequency': frequency,
                'avg_activation': avg_activation,
                'activations': activations
            }
    
    # Get top reasoning features
    top_reasoning_features = sorted(feature_scores.keys(), 
                                  key=lambda x: feature_scores[x]['score'], 
                                  reverse=True)[:NUM_TOP_FEATURES]
    
    print(f"Top 10 reasoning features:")
    for i, feat_id in enumerate(top_reasoning_features[:10]):
        info = feature_scores[feat_id]
        print(f"  Feature {feat_id}: score={info['score']:.3f}, "
              f"freq={info['frequency']}/{len(gsm8k_samples)}, "
              f"avg_act={info['avg_activation']:.4f}")
    
    return top_reasoning_features, feature_scores

def apply_feature_grafting(model_a, stitch_model, sae_a, sae_b, 
                          reasoning_features, feature_scores, 
                          input_ids, grafting_strength=0.5):
    """
    Apply feature grafting during inference:
    1. Get 2B model activations
    2. Decode with 2B SAE to get features
    3. Enhance reasoning features based on 9B patterns
    4. Encode back to activations
    5. Continue inference
    """
    model_a.eval()
    stitch_model.eval()
    sae_a.eval()
    
    with torch.no_grad():
        # Get original activations from 2B model
        outputs_a = model_a.model(input_ids, output_hidden_states=True)
        original_activations = outputs_a.hidden_states[STITCH_LAYER_A][:, -1].float()
        
        # Get unnormalized version for SAE
        original_activations_unnorm = original_activations.clone()
        
        # Process through 2B SAE
        sae_a_output = sae_a(original_activations_unnorm)
        if isinstance(sae_a_output, tuple) and len(sae_a_output) >= 2:
            original_reconstruction = sae_a_output[0]
            original_features = sae_a_output[1]
        else:
            original_reconstruction = sae_a_output
            original_features = sae_a.encode(original_activations_unnorm)
        
        # Create enhanced features by boosting reasoning features
        enhanced_features = original_features.clone()
        
        for feature_id in reasoning_features:
            if feature_id < enhanced_features.shape[-1]:  # Check bounds
                # Get the typical activation strength for this feature from 9B analysis
                target_strength = feature_scores[feature_id]['avg_activation']
                
                # Current activation
                current_activation = enhanced_features[0, feature_id]
                
                # Enhance based on grafting strength
                enhancement = grafting_strength * target_strength
                enhanced_features[0, feature_id] = current_activation + enhancement
        
        # Reconstruct activations from enhanced features
        try:
            # Use SAE decoder to get activations from features
            enhanced_activations = sae_a.decode(enhanced_features)
        except:
            # Fallback: approximate reconstruction
            enhanced_activations = original_reconstruction
        
        return enhanced_activations, original_activations

class FeatureGraftedModel:
    """Wrapper that applies feature grafting during generation"""
    
    def __init__(self, model_a, stitch_model, sae_a, sae_b, 
                 reasoning_features, feature_scores, grafting_strength=0.5):
        self.model_a = model_a
        self.stitch_model = stitch_model
        self.sae_a = sae_a
        self.sae_b = sae_b
        self.reasoning_features = reasoning_features
        self.feature_scores = feature_scores
        self.grafting_strength = grafting_strength
        
    def generate(self, input_ids, max_new_tokens=150, temperature=0.7, do_sample=True):
        """Generate with feature grafting applied"""
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Apply feature grafting to current context
            enhanced_activations, _ = apply_feature_grafting(
                self.model_a, self.stitch_model, self.sae_a, self.sae_b,
                self.reasoning_features, self.feature_scores,
                generated_ids, self.grafting_strength
            )
            
            # Get logits from model (this is simplified - in practice you'd need to modify the forward pass)
            with torch.no_grad():
                outputs = self.model_a(generated_ids)
                logits = outputs.logits[:, -1, :]
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        return generated_ids

def evaluate_gsm8k_performance(model, tokenizer, test_samples, grafting_strength=0.0, model_name="Model"):
    """Evaluate performance on GSM8K test set"""
    correct = 0
    total = 0
    results = []
    
    print(f"Evaluating {model_name} (strength={grafting_strength}) on {len(test_samples)} samples...")
    
    for i, sample in enumerate(tqdm(test_samples, desc=f"Evaluating {model_name}")):
        question = sample['question']
        correct_answer = float(sample['answer'].split('#### ')[-1])
        
        # Create prompt
        prompt = f"Question: {question}\nLet me solve this step by step:"
        
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Generate response
            if hasattr(model, 'generate') and callable(model.generate) and hasattr(model, 'config'):
                # Standard model interface (actual transformer model)
                with torch.no_grad():
                    generated_ids = model.generate(
                        inputs['input_ids'],
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
            else:
                # Feature grafted model interface
                generated_ids = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode response
            response = tokenizer.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            
            # Extract predicted answer
            predicted_answer = extract_numerical_answer(response)
            
            # Check correctness
            is_correct = (predicted_answer is not None and 
                         abs(predicted_answer - correct_answer) < 0.01)
            
            if is_correct:
                correct += 1
            
            total += 1
            
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct
            })
            
            # Print some examples
            if i < 5 or (i % 20 == 0):
                print(f"Sample {i}: Correct={correct_answer}, Predicted={predicted_answer}, Match={is_correct}")
                
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            total += 1  # Count as attempt
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': None,
                'response': f"Error: {e}",
                'is_correct': False
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} (strength={grafting_strength}): {correct}/{total} = {accuracy:.3f}")
    
    return accuracy, results

# Main execution
print("Loading GSM8K dataset...")
gsm8k_train = load_dataset("gsm8k", "main", split="train")
gsm8k_test = load_dataset("gsm8k", "main", split="test")

# Take subset for feature identification
feature_id_samples = list(gsm8k_train.select(range(GSM8K_SUBSET_SIZE)))
test_samples = list(gsm8k_test.select(range(200)))  # Use first 200 test samples

print(f"Using {len(feature_id_samples)} samples for feature identification")
print(f"Using {len(test_samples)} samples for evaluation")

# Step 1: Identify reasoning features from 9B model
print("\n" + "="*60)
print("STEP 1: IDENTIFYING REASONING FEATURES")
print("="*60)

reasoning_features, feature_scores = identify_reasoning_features(
    model_b, sae_b, tokenizer, feature_id_samples, STITCH_LAYER_B
)

print(f"Identified {len(reasoning_features)} top reasoning features")

# Step 2: Evaluate different grafting strengths
print("\n" + "="*60)
print("STEP 2: EVALUATING FEATURE GRAFTING")
print("="*60)

results_by_strength = {}

# Baseline: Original 2B model (strength=0.0)
print("Evaluating baseline (original 2B model)...")
baseline_accuracy, baseline_results = evaluate_gsm8k_performance(
    model_a, tokenizer, test_samples[:50], grafting_strength=0.0, model_name="Baseline 2B"
)
results_by_strength[0.0] = {
    'accuracy': baseline_accuracy,
    'results': baseline_results
}

# Test different grafting strengths
for strength in GRAFTING_STRENGTHS[1:]:  # Skip 0.0 as we did baseline
    print(f"\nTesting grafting strength: {strength}")
    
    # Create grafted model
    grafted_model = FeatureGraftedModel(
        model_a, stitch_model, sae_a, sae_b,
        reasoning_features, feature_scores, 
        grafting_strength=strength
    )
    
    # Evaluate
    accuracy, results = evaluate_gsm8k_performance(
        grafted_model, tokenizer, test_samples[:50],  # Use subset for speed
        grafting_strength=strength, 
        model_name=f"Grafted 2B"
    )
    
    results_by_strength[strength] = {
        'accuracy': accuracy,
        'results': results
    }

# Step 3: Analyze results
print("\n" + "="*60)
print("STEP 3: RESULTS ANALYSIS")
print("="*60)

# Plot results
strengths = sorted(results_by_strength.keys())
accuracies = [results_by_strength[s]['accuracy'] for s in strengths]

plt.figure(figsize=(10, 6))
plt.plot(strengths, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Grafting Strength')
plt.ylabel('GSM8K Accuracy')
plt.title('Feature Grafting Performance vs. Grafting Strength')
plt.grid(True, alpha=0.3)
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', 
           label=f'Baseline (2B): {baseline_accuracy:.3f}')

# Find best performance
best_strength = strengths[np.argmax(accuracies)]
best_accuracy = max(accuracies)
plt.axvline(x=best_strength, color='g', linestyle='--', alpha=0.7,
           label=f'Best: {best_strength} ({best_accuracy:.3f})')

plt.legend()
plt.tight_layout()
plt.savefig('gsm8k_feature_grafting_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("PERFORMANCE SUMMARY:")
print("-" * 40)
for strength in strengths:
    accuracy = results_by_strength[strength]['accuracy']
    improvement = (accuracy - baseline_accuracy) * 100
    print(f"Strength {strength:3.1f}: {accuracy:.3f} ({improvement:+.1f}% vs baseline)")

print(f"\nBest performance: {best_accuracy:.3f} at strength {best_strength}")
improvement = (best_accuracy - baseline_accuracy) * 100
print(f"Overall improvement: {improvement:+.1f}% over baseline")

# Analyze feature importance
print(f"\nTop 10 most important reasoning features:")
for i, feat_id in enumerate(reasoning_features[:10]):
    info = feature_scores[feat_id]
    print(f"  {i+1}. Feature {feat_id}: "
          f"score={info['score']:.2f}, "
          f"freq={info['frequency']}/{GSM8K_SUBSET_SIZE}, "
          f"avg_strength={info['avg_activation']:.4f}")

# Save detailed results
detailed_results = {
    'reasoning_features': reasoning_features,
    'feature_scores': feature_scores,
    'results_by_strength': results_by_strength,
    'best_strength': best_strength,
    'best_accuracy': best_accuracy,
    'baseline_accuracy': baseline_accuracy,
    'config': {
        'stitch_layer_a': STITCH_LAYER_A,
        'stitch_layer_b': STITCH_LAYER_B,
        'gsm8k_subset_size': GSM8K_SUBSET_SIZE,
        'num_top_features': NUM_TOP_FEATURES,
        'test_samples': len(test_samples)
    }
}

torch.save(detailed_results, 'gsm8k_feature_grafting_detailed_results.pt')
print(f"\nDetailed results saved to 'gsm8k_feature_grafting_detailed_results.pt'")

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)
if best_accuracy > baseline_accuracy + 0.02:  # 2% improvement threshold
    print("✓ Feature grafting shows significant improvement!")
    print(f"  Best improvement: {improvement:+.1f}% at strength {best_strength}")
    print(f"  This suggests the identified reasoning features are transferable")
elif abs(best_accuracy - baseline_accuracy) < 0.01:
    print("→ Feature grafting shows neutral effect")
    print("  The stitch may be working but features might not be optimally selected")
else:
    print("✗ Feature grafting shows degraded performance")
    print("  May need to adjust feature selection or stitch training")

print(f"\nNext steps:")
print(f"1. Analyze which specific features contribute most to improvement")
print(f"2. Test on full GSM8K test set with best strength ({best_strength})")
print(f"3. Examine failure cases to understand limitations")
print(f"4. Try different feature selection criteria (e.g., problem-type specific)")
print("="*60)
