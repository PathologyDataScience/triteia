"""
Plot regular inference, inference without IO and  Multiuser performance with and without TRT
"""
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Extracted from bs128.
# Not reading from tensorboards for this plot just to save some time and complexity
data = '''
model_name,gpu_count,gpu_throughput,cpu_throughput
ResNet-50,1,1475,1285
ResNet-50,2,2606,1749
ResNet-50,4,2814,1889
ResNet-50,6,2947,1841
ResNet-50,8,2830,1831
UNI,1,0675,0570
UNI,2,1310,1022
UNI,4,2279,1539
UNI,6,2425,1829
UNI,8,2392,1853
Prov-GigaPath,1,0212,0202
Prov-GigaPath,2,0429,0407
Prov-GigaPath,4,0844,0792
Prov-GigaPath,6,1252,1092
Prov-GigaPath,8,1602,1292
'''


# Colors using colorblind palette
palette = sns.color_palette("colorblind", n_colors=3)
colors = {
    # f"CPU-Based": palette[0],
    # f"GPU-Based": palette[1],
    f"CPU-Based": "#1f77b4",
    f"GPU-Based": "#ff7f0e",
}

# Parse the data
df = pd.read_csv(pd.io.common.StringIO(data))

# Define the order of models: Prov-Gigapath, UNI, then ResNet-50
model_order = ["Prov-GigaPath", "UNI", "ResNet-50"]

# Create figure with 3 subplots (one for each model)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw = {"hspace": 0.5})
fig.suptitle("CPU- vs GPU-based Preprocessing Throughput Comparison", fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.82)  # Add vertical space after the suptitle

# GPU counts for x-axis
gpu_counts = [1, 2, 4, 6, 8]
x_positions = range(len(gpu_counts))
bar_width = 0.35

label_idx = 0

# Plot each model in its own subplot
for idx, model in enumerate(model_order):
    ax = axes[idx]
    
    # Filter data for this model
    model_data = df[df['model_name'] == model].sort_values('gpu_count')
    
    # Extract values for plotting
    gpu_throughput = model_data['gpu_throughput'].values
    cpu_throughput = model_data['cpu_throughput'].values
    
    # Create bars for GPU and CPU
    x_pos = list(x_positions)
    ax.bar([x - bar_width/2 for x in x_pos], cpu_throughput, bar_width, 
           label='CPU-Based', color=colors['CPU-Based'], edgecolor = "black", linewidth = 0.5)
    ax.bar([x + bar_width/2 for x in x_pos], gpu_throughput, bar_width,
           label='GPU-Based', color=colors['GPU-Based'], edgecolor = "black", linewidth = 0.5)
    
    # Set labels and title
    ax.set_xlabel('GPUs', fontsize=16)
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gpu_counts, fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    letter = string.ascii_lowercase[label_idx]
    ax.text(
        0.09,
        1.02,
        f"{letter})",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    label_idx += 1

    # Only show y-axis label and ticks on the first subplot
    if idx == 0:
        ax.set_ylabel('tiles/s', fontsize=16)
    else:
        ax.set_yticklabels([])

# Set the same y-axis limits for all subplots
all_throughputs = list(df['gpu_throughput']) + list(df['cpu_throughput'])
y_max = max(all_throughputs) * 1.1  # Add 10% margin
for ax in axes:
    ax.set_ylim(0, y_max)

# Create a single legend below the figure
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=colors['CPU-Based'], label='CPU-Based'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['GPU-Based'], label='GPU-Based'),
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, fontsize=18)

plt.tight_layout()

figure_dst = "plot_out/cpu-vs-gpu.png"
plt.savefig(figure_dst, dpi=300, bbox_inches="tight")
    
# Also save as SVG
figure_dst_svg = "plot_out/cpu-vs-gpu.svg"
plt.savefig(figure_dst_svg, format='svg', bbox_inches="tight")
    
# plt.show()
plt.close()
print(f"Saved figure to {figure_dst}")
print(f"Saved figure to {figure_dst_svg}")

print("\n=== Percentage Difference Analysis (GPU vs CPU) ===\n")
for model in model_order:
    model_data = df[df['model_name'] == model].sort_values('gpu_count')
    print(f"{model}:")
    for _, row in model_data.iterrows():
        gpu_count = row['gpu_count']
        gpu_throughput = row['gpu_throughput']
        cpu_throughput = row['cpu_throughput']
        
        # Calculate percentage difference: (GPU - CPU) / CPU * 100
        pct_diff = ((gpu_throughput - cpu_throughput) / cpu_throughput) * 100
        
        print(f"  GPU Count {int(gpu_count)}: {pct_diff:+.1f}% (GPU: {gpu_throughput}, CPU: {cpu_throughput})")
    print()
