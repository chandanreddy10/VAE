import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = "logs"
csv_files = sorted([file for file in os.listdir(LOGS_DIR) if file.endswith(".csv")])

COLUMNS = ["train_loss", "train_recon", "train_kl", "test_loss", "test_recon", "test_kl"]

cmap = plt.get_cmap('gist_rainbow', len(csv_files))
file_colors = {file: cmap(i) for i, file in enumerate(csv_files)}

fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axs = axs.flatten()

global_handles = []
global_labels = []

for i, col in enumerate(COLUMNS):
    ax = axs[i]
    
    for file in csv_files:
        file_path = os.path.join(LOGS_DIR, file)
        try:
            df = pd.read_csv(file_path)
            x_axis = df['epoch'] if 'epoch' in df.columns else df.index
            label_name = os.path.splitext(file)[0]
     
            line, = ax.plot(
                x_axis, 
                df[col], 
                color=file_colors[file], 
                alpha=0.75, 
                linewidth=1.2, 
                label=label_name
            )

            if i == 0:
                global_handles.append(line)
                global_labels.append(label_name)
                
        except KeyError:
            continue  
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    ax.set_title(f'Metric: {col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle=':', alpha=0.5)


axs[4].set_xlabel('Epoch / Steps')
axs[5].set_xlabel('Epoch / Steps')


fig.legend(
    global_handles, 
    global_labels, 
    loc='center left', 
    bbox_to_anchor=(1.02, 0.5), 
    ncol=2, 
    fontsize='xx-large', 
    title="Log Files Runs",
    title_fontproperties={'weight':'bold'}
)


plt.tight_layout()

plt.savefig('all_logs_clean_comparison.png', dpi=300, bbox_inches='tight')
plt.show()