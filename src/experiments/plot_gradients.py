import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Idea is to Combine all gradient JSON files and plot the gradients.
JSON_DIR = "logs" 
all_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))

data_list = []

for file_path in all_files:
    run_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r') as f:
        run_data = json.load(f)
        
    for epoch_key, layers in run_data.items():
        # "epoch_1" -> 1
        try:
            epoch = int(epoch_key.split('_')[1])
        except IndexError:
            epoch = int(epoch_key)
            
        for layer_name, metrics in layers.items():
            data_list.append({
                "run": run_name,
                "epoch": epoch,
                "layer": layer_name,
                "norm": metrics.get("norm", 0.0),
                "dead_pct": metrics.get("dead_pct", 0.0)
            })


df = pd.DataFrame(data_list)
print(f"Loaded {df['run'].nunique()} runs across {df['epoch'].nunique()} epochs.")

# Group by layer and epoch, then average norm
heatmap_data = df.groupby(['layer', 'epoch'])['norm'].mean().unstack(level='epoch')

# Encoder -> Bottleneck -> Decoder
with open(all_files[0], 'r') as f:
    sample_data = json.load(f)
layer_order = list(next(iter(sample_data.values())).keys())
heatmap_data = heatmap_data.reindex(layer_order)

plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='viridis', norm=plt.cm.colors.LogNorm(), cbar_kws={'label': 'Mean Gradient Norm (Log Scale)'})
plt.title("Architecture Gradient Flow across 100 Epochs (Averaged over 30 Runs)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Network Layers / Parameters")
plt.tight_layout()
plt.savefig("gradient_flow_heatmap.png", dpi=300)
plt.show()

# Group by layer and epoch for dead percentage
heatmap_dead = df.groupby(['layer', 'epoch'])['dead_pct'].mean().unstack(level='epoch')
heatmap_dead = heatmap_dead.reindex(layer_order)

plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_dead, cmap='Reds', vmin=0, vmax=100, cbar_kws={'label': 'Dead Neuron Percentage (%)'})
plt.title("Dead Neuron Percentage Over Time per Layer", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Network Layers / Parameters")
plt.tight_layout()
plt.savefig("dead_neurons_heatmap.png", dpi=300)
plt.show()

target_layer = "mu_logvar.weight"
layer_df = df[df['layer'] == target_layer]

plt.figure(figsize=(12, 6))

sns.lineplot(data=layer_df, x='epoch', y='norm', hue='run', palette='tab20', alpha=0.7, linewidth=1.5)

plt.title(f"Gradient Norm History Across All 30 Runs for Layer: {target_layer}", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.grid(True, linestyle=":", alpha=0.5)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small', title="Run Files")
plt.tight_layout()
plt.savefig("single_layer_trajectory.png", dpi=300, bbox_inches='tight')
plt.show()