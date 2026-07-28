"""
Plot regular inference, inference without IO and  Multiuser performance with and without TRT
"""
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Load data
multiuser_df = pd.read_csv("plot_out/multiuser_scaling.csv")
inferenceonly_df = pd.read_csv("plot_out/tritoninferenceonlygpu_scaling.csv")
raw_df = pd.read_csv("plot_out/tritongpu_scaling.csv")

# Models to plot
models_config = [
    {
        "name": "Prov-GigaPath",
        "trt_pattern": "Prov-GigaPath-trt",
        "base_pattern_inf": "tritonProv-Gigapath-trtinferenceonly",
        "base_pattern": "tritonProv-GigaPath-trt",
        "non_trt_pattern": "tritonProv-GigaPath",
        "non_trt_pattern_inf": "tritonProv-GigaPathinferenceonly",
        "multiuser_name": "Prov-GigaPath_trt",
        "multiuser_name_non_trt": "Prov-GigaPath",
    },
    {
        "name": "ResNet-50",
        "trt_pattern": "ResNet-50-trt",
        "base_pattern_inf": "tritonResNet-50-trtinferenceonly",
        "base_pattern": "tritonResNet-50-trt",
        "non_trt_pattern": "tritonResNet-50",
        "non_trt_pattern_inf": "tritonResNet-50inferenceonly",
        "multiuser_name": "ResNet-50-trt",
        "multiuser_name_non_trt": "ResNet-50",
    },
]

gpu_values = [1, 8]
batch_size = 256
concurrency = 10

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# Colors using colorblind palette
palette = sns.color_palette("colorblind", n_colors=3)
colors = {
    f"With IO": palette[0],
    f"Without IO": palette[1],
    f"Multiuser": palette[2],
}

bar_width = 0.25
x_positions = {1: 0, 8: 1}

label_idx = 0
for ax_idx, model_cfg in enumerate(models_config):
    ax = axes[ax_idx]

    # --- TRT bar data (raw = with I/O) ---
    trt_data = raw_df[
        (raw_df["model"].str.contains(model_cfg["trt_pattern"]))
        & (raw_df["batch_size"] == batch_size)
        & (raw_df["gpus"].isin(gpu_values))
    ].copy()

    # --- Non-TRT baseline for TRT bar (raw = with I/O) ---
    non_trt_data = raw_df[
        (raw_df["model"] == model_cfg["non_trt_pattern"])
        & (raw_df["batch_size"] == batch_size)
        & (raw_df["gpus"].isin(gpu_values))
    ].copy()

    # --- Inferenceonly bar data (TRT, inference-only) ---
    trt_data_inf = inferenceonly_df[
        (inferenceonly_df["model"].str.contains(model_cfg["trt_pattern"]))
        & (inferenceonly_df["batch_size"] == batch_size)
        & (inferenceonly_df["gpus"].isin(gpu_values))
    ].copy()

    # --- Non-TRT baseline for Inferenceonly bar ---
    non_trt_data_inf = inferenceonly_df[
        (inferenceonly_df["model"] == model_cfg["non_trt_pattern_inf"])
        & (inferenceonly_df["batch_size"] == batch_size)
        & (inferenceonly_df["gpus"].isin(gpu_values))
    ].copy()

    # --- Multiuser bar data (TRT) ---
    multiuser_data = multiuser_df[
        (multiuser_df["model"] == model_cfg["multiuser_name"])
        & (multiuser_df["concurrency"] == concurrency)
        & (multiuser_df["gpus"].isin(gpu_values))
    ].copy()

    # --- Non-TRT baseline for Multiuser bar ---
    multiuser_data_non_trt = multiuser_df[
        (multiuser_df["model"] == model_cfg["multiuser_name_non_trt"])
        & (multiuser_df["concurrency"] == concurrency)
        & (multiuser_df["gpus"].isin(gpu_values))
    ].copy()

    # Categories, bar data, and corresponding non-TRT baseline data
    categories = [
        f"With IO",
        f"Without IO",
        f"Multiuser",
    ]
    bar_data = [
        (trt_data, "throughput_mean"),
        (trt_data_inf, "throughput_mean"),
        (multiuser_data, "throughput_agg_mean"),
    ]
    baseline_data = [
        (non_trt_data, "throughput_mean"),
        (non_trt_data_inf, "throughput_mean"),
        (multiuser_data_non_trt, "throughput_agg_mean"),
    ]

    # Track bar x positions per category for hline overlay
    cat_x_vals = {}

    for cat_idx, (category, (df, col)) in enumerate(zip(categories, bar_data)):
        heights = []
        x_vals = []
        for gpu in gpu_values:
            gpu_data = df[df["gpus"] == gpu]
            if not gpu_data.empty:
                heights.append(float(gpu_data[col].iloc[0]))
            else:
                heights.append(0)
            x_vals.append(x_positions[gpu] + (cat_idx - 1) * bar_width)

        cat_x_vals[category] = x_vals

        ax.bar(
            x_vals,
            heights,
            bar_width,
            label=category,
            color=colors[category],
            edgecolor="black",
            linewidth=0.5,
        )

    # Draw non-TRT Python baseline as black line inside each bar
    for cat_idx, (category, (bl_df, bl_col)) in enumerate(
        zip(categories, baseline_data)
    ):
        for gpu_idx, gpu in enumerate(gpu_values):
            bl_gpu = bl_df[bl_df["gpus"] == gpu]
            if not bl_gpu.empty:
                baseline_value = float(bl_gpu[bl_col].iloc[0])
                bar_x = cat_x_vals[category][gpu_idx]
                ax.hlines(
                    y=baseline_value,
                    xmin=bar_x - bar_width / 2,
                    xmax=bar_x + bar_width / 2,
                    colors="black",
                    linestyles="solid",
                    linewidth=2.5,
                )

    # Add subplot letter label (a, b, c, ...)
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

    ax.set_xlabel("GPUs", fontsize=14)
    if ax_idx == 0:
        ax.set_ylabel("tiles / second", fontsize=14)
    ax.set_title(model_cfg["name"], fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1", "8"], fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

# Shared legend at the bottom (including the baseline line)
handles, labels = axes[0].get_legend_handles_labels()
baseline_handle = Line2D(
    [0], [0], color="black", linestyle="solid", linewidth=2.5, label="Python backend"
)
handles.append(baseline_handle)
labels.append("Python backend")

fig.suptitle(
    f"Throughput scaling with TRT backend",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
#  TRT vs Inferenceonly vs Multiuser (c=10)\nBatch Size {batch_size}, GPUs 1 & 8
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=True,
    fontsize=11,
)

figure_dst = "plot_out/trt_comparison.png"
plt.savefig(figure_dst, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()
print(f"Saved figure to {figure_dst}")
