from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
import json
from latent_control.vector_diagnostic import ControlVectorFeasibilityAnalyzer

# Load config
with open('configs/production.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model from config
import warnings
warnings.filterwarnings('ignore', message='`torch_dtype` is deprecated')

model = AutoModelForCausalLM.from_pretrained(
    config['model']['model_path'],
    torch_dtype=torch.float16,
    device_map=config['model']['device']
)
tokenizer = AutoTokenizer.from_pretrained(config['model']['model_path'])

# Load your prompts from files
def load_prompts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

high_entropy_prompts = load_prompts('prompts/high_entropy_generated.txt')
low_entropy_prompts = load_prompts('prompts/low_entropy_generated.txt')

# Initialize analyzer
analyzer = ControlVectorFeasibilityAnalyzer(model, tokenizer)

# Calculate layer range from config's layer_fraction
n_layers = model.config.num_hidden_layers
target_layer = int(n_layers * config['model']['layer_fraction'])
layer_range = (max(0, target_layer - 5), min(n_layers, target_layer + 5))

print(f"Analyzing layers {layer_range[0]} to {layer_range[1]} around target layer {target_layer}")

# Run analysis
layer_results = analyzer.analyze_layer_suitability(
    high_entropy_prompts,
    low_entropy_prompts,
    layer_range=layer_range,
    position='last'
)

# Generate and print report
report = analyzer.generate_report(layer_results)
print(report)

# Save results
output_path = f"{config['model']['cache_dir']}/feasibility_analysis.json"
with open(output_path, 'w') as f:
    # Convert tensors/arrays to lists for JSON serialization
    serializable_results = {
        k: {metric: (v.tolist() if hasattr(v, 'tolist') else v)
            for metric, v in metrics.items()}
        for k, metrics in layer_results.items()
    }
    json.dump(serializable_results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Find best layer
best_layer = max(layer_results.items(), 
                 key=lambda x: x[1].get('lda_accuracy', 0) or 0)[0]

print(f"\n{'='*80}")
print(f"OUTLIER ANALYSIS FOR LAYER {best_layer}")
print(f"{'='*80}\n")

# Identify outliers/misclassified prompts
outliers = analyzer.identify_outliers(
    high_entropy_prompts,
    low_entropy_prompts,
    layer_idx=best_layer
)

print("Low entropy prompts that cluster with high entropy:")
if outliers['misclassified_low_entropy']:
    for item in outliers['misclassified_low_entropy']:
        print(f"  [{item['index']}] {item['prompt']}")
else:
    print("  None - all low entropy prompts are correctly separated!")

print("\nHigh entropy prompts that cluster with low entropy:")
if outliers['misclassified_high_entropy']:
    for item in outliers['misclassified_high_entropy']:
        print(f"  [{item['index']}] {item['prompt']}")
else:
    print("  None - all high entropy prompts are correctly separated!")

# Save outliers to file
outliers_path = f"{config['model']['cache_dir']}/outliers_layer_{best_layer}.json"
with open(outliers_path, 'w') as f:
    json.dump(outliers, f, indent=2)
print(f"\nOutliers saved to {outliers_path}")

# Visualize best layer with labels
print(f"\nGenerating labeled visualization for layer {best_layer}...")
analyzer.visualize_activation_space_with_labels(
    high_entropy_prompts,
    low_entropy_prompts,
    layer_idx=best_layer,
    save_path=f"{config['model']['cache_dir']}/layer_{best_layer}_labeled_viz.png",
    max_label_length=50
)

# Also create the original unlabeled version
analyzer.visualize_activation_space(
    high_entropy_prompts,
    low_entropy_prompts,
    layer_idx=best_layer,
    save_path=f"{config['model']['cache_dir']}/layer_{best_layer}_viz.png"
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Check the '{config['model']['cache_dir']}' directory for:")
print("  - feasibility_analysis.json (full metrics)")
print(f"  - outliers_layer_{best_layer}.json (misclassified prompts)")
print(f"  - layer_{best_layer}_viz.png (unlabeled visualization)")
print(f"  - layer_{best_layer}_labeled_viz.png (labeled visualization)")
