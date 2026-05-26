#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Read tensorboarddata from benchmarks and write them to CSV files.
This way it is easy to modify the plot code without having to parse all the tensorboards.
"""

import argparse
import glob
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

GPU_SCALING_CSV = "gpu_scaling_ramin.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tensorboard data")
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        nargs="+",
        help="One or more directories containing tensorboard runs (or parent directories of runs)",
        required=True,
        default=["."],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory for plot files",
        default="./plot_out",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for d in args.tensorboard_dir:
        if not os.path.exists(d) or not os.path.isdir(d):
            raise ValueError(
                f"Tensorboard directory '{d}' does not exist or is not a directory"
            )

    return args


def extract_tensorboard_data(logdir: str, plot_key: str) -> pd.DataFrame:
    """
    Read scalar summaries for `plot_key` from a TensorBoard logdir.

    Returns a dataframe with columns: step, wall_time, value.

    Note: TensorFlow is imported lazily here so that if gpu_scaling.csv exists,
    the program can start quickly without importing TensorFlow at all.
    """
    import tensorflow as tf

    event_files = glob.glob(
        os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True
    )
    if not event_files:
        logging.warning("No event files found under %s", logdir)
        return pd.DataFrame(columns=["step", "wall_time", "value"])

    rows = []
    for ef in sorted(event_files):
        try:
            for e in tf.compat.v1.train.summary_iterator(ef):
                if not hasattr(e, "summary") or e.summary is None:
                    raise ValueError(
                        f"Event file {ef} does not contain a valid summary"
                    )

                for v in e.summary.value:
                    if v.tag != plot_key:
                        continue
                    # Prefer simple_value when present
                    if hasattr(v, "simple_value"):
                        value = float(v.simple_value)
                    else:
                        # Some summaries store tensor; skip if we can't read it robustly here
                        continue
                    rows.append(
                        {
                            "step": int(getattr(e, "step", 0)),
                            "wall_time": float(getattr(e, "wall_time", 0.0)),
                            "value": value,
                        }
                    )
        except Exception as dle:
            logging.warning(
                "Skipping event file %s due to data loss error: %s", ef, dle
            )

    if not rows:
        return pd.DataFrame(columns=["step", "wall_time", "value"])

    df = pd.DataFrame(rows).sort_values(["step", "wall_time"]).reset_index(drop=True)
    return df


def _candidate_run_dirs(tensorboard_dirs):
    # Collect candidate run directories from the provided list (roots + direct children)
    candidate_run_dirs = []
    for root in tensorboard_dirs:
        candidate_run_dirs.append(root)
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                candidate_run_dirs.append(p)
    return sorted(set(candidate_run_dirs))


def build_gpu_scaling_dataframe(
    tensorboard_dirs, plot_key: str, run_prefix="", run_suffix=""
) -> pd.DataFrame:
    """
    Scan tensorboard directories and build a single dataframe of run-level summaries.

    Output columns:
      model, run, logdir, gpus, batch_size, plot_key, throughput_mean, throughput_last, gpu_util_percent_mean

    If run_prefix is non-empty, the run dir must start with that prefix, e.g. "pytorch".
    If run_suffix is non-empty, the run dir must end with that suffix, e.g. "inferenceonlytriton".
    """
    prefix = re.escape(run_prefix)
    suffix = re.escape(run_suffix)

    model_specs = [
        {
            "name": (
                f"{prefix}Prov-GigaPath{suffix}"
                if (run_prefix or run_suffix)
                else "GigaPath"
            ),
            "run_dir_re": re.compile(rf"{prefix}gigapathgpu[1-8]bs[0-9]+{suffix}$"),
            "parse_re": re.compile(
                rf"{prefix}gigapathgpu(?P<gpus>[1-8])bs(?P<bs>[0-9]+){suffix}$"
            ),
        },
        {
            "name": (f"{prefix}UNI{suffix}" if (run_prefix or run_suffix) else "UNI"),
            "run_dir_re": re.compile(rf"{prefix}unigpu[1-8]bs[0-9]+{suffix}$"),
            "parse_re": re.compile(
                rf"{prefix}unigpu(?P<gpus>[1-8])bs(?P<bs>[0-9]+){suffix}$"
            ),
        },
        {
            "name": (
                f"{prefix}ResNet-50{suffix}"
                if (run_prefix or run_suffix)
                else "ResNet-50"
            ),
            "run_dir_re": re.compile(rf"{prefix}resnet50gpu[1-8]bs[0-9]+{suffix}$"),
            "parse_re": re.compile(
                rf"{prefix}resnet50gpu(?P<gpus>[1-8])bs(?P<bs>[0-9]+){suffix}$"
            ),
        },
        {
            "name": (
                f"{prefix}ResNet-50-trt{suffix}"
                if (run_prefix or run_suffix)
                else "ResNet-50-trt"
            ),
            "run_dir_re": re.compile(
                rf"{prefix}resnet50_trt_uint8gpu[1-8]bs[0-9]+{suffix}$"
            ),
            "parse_re": re.compile(
                rf"{prefix}resnet50_trt_uint8gpu(?P<gpus>[1-8])bs(?P<bs>[0-9]+){suffix}$"
            ),
            # tritonresnet50_trt_uint8gpu2bs64
        },
        {
            "name": (
                f"{prefix}Prov-GigaPath-trt{suffix}"
                if (run_prefix or run_suffix)
                else "Prov-GigaPath-trt"
            ),
            "run_dir_re": re.compile(
                rf"{prefix}gigapath_trt_uint8gpu[1-8]bs[0-9]+{suffix}$"
            ),
            "parse_re": re.compile(
                rf"{prefix}gigapath_trt_uint8gpu(?P<gpus>[1-8])bs(?P<bs>[0-9]+){suffix}$"
            ),
        },
    ]

    candidate_run_dirs = _candidate_run_dirs(tensorboard_dirs)

    rows = []
    for spec in model_specs:
        run_dirs = []
        for d in candidate_run_dirs:
            base = os.path.basename(os.path.normpath(d))
            if spec["run_dir_re"].match(base):
                run_dirs.append(d)

        if not run_dirs:
            logging.warning(
                "No tensorboard run directories matched for model '%s' (prefix=%r, suffix=%r)",
                spec["name"],
                run_prefix,
                run_suffix,
            )
            continue
        for d in run_dirs:
            base = os.path.basename(os.path.normpath(d))
            m = spec["parse_re"].match(base)
            if not m:
                continue

            gpus = int(m.group("gpus"))
            bs = int(m.group("bs"))

            df = extract_tensorboard_data(d, plot_key=plot_key)
            if df.empty:
                logging.warning("No data for key '%s' in %s", plot_key, d)
                continue

            # Extract GPU utilization data — average across all GPUs in this run
            gpu_util_means = []
            for gpu_idx in range(gpus):
                gpu_util_key = f"gpu_gpu_{gpu_idx}_util_percent"
                gpu_util_df = extract_tensorboard_data(d, plot_key=gpu_util_key)
                if not gpu_util_df.empty:
                    gpu_util_means.append(float(gpu_util_df["value"].mean()))
                else:
                    logging.warning("No data for key '%s' in %s", gpu_util_key, d)

            gpu_util_mean = (
                float(sum(gpu_util_means) / len(gpu_util_means))
                if gpu_util_means
                else None
            )

            rows.append(
                {
                    "model": spec["name"],
                    "run": base,
                    "logdir": d,
                    "gpus": gpus,
                    "batch_size": bs,
                    "plot_key": plot_key,
                    "throughput_mean": float(df["value"].mean()),
                    "throughput_last": float(df["value"].iloc[-1]),
                    "gpu_util_percent_mean": gpu_util_mean,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "model",
            "run",
            "logdir",
            "gpus",
            "batch_size",
            "plot_key",
            "throughput_mean",
            "throughput_last",
            "gpu_util_percent_mean",
        ],
    )

    if not out.empty:
        out = out.sort_values(["model", "batch_size", "gpus", "run"]).reset_index(
            drop=True
        )
    return out


def load_or_build_gpu_scaling(
    tensorboard_dirs,
    csv_path: str,
    plot_key: str,
    run_prefix="triton",
    run_suffix="",
) -> pd.DataFrame:
    """
    If csv exists, load it and skip reading TensorBoards.
    Otherwise, build dataframe from TensorBoards and write csv.
    """
    if os.path.exists(csv_path) and os.path.isfile(csv_path):
        logging.info(
            "Found %s; loading cached data (skipping TensorBoard read).", csv_path
        )
        df = pd.read_csv(csv_path)
        return df

    logging.info("%s not found; reading TensorBoards and creating cache.", csv_path)
    df = build_gpu_scaling_dataframe(
        tensorboard_dirs,
        plot_key=plot_key,
        run_prefix=run_prefix,
        run_suffix=run_suffix,
    )
    if df.empty:
        raise ValueError(
            f"No matching runs with scalar data found for plot key '{plot_key}' when requesting {csv_path} from {tensorboard_dirs}."
        )

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)
    return df


def read_and_write_data(tensorboard_dirs, output_dir):
    plot_key = "throughput_total_tiles_per_second"

    for run_prefix in ["triton"]:
        for run_suffix in ["", ": pre-loaded slides"]:
            key = "inferenceonly" if run_suffix else ""
            df = load_or_build_gpu_scaling(
                tensorboard_dirs=tensorboard_dirs,
                csv_path=os.path.join(
                    output_dir, f"{run_prefix}{key}{GPU_SCALING_CSV}"
                ),
                plot_key=plot_key,
                run_prefix=run_prefix,
                run_suffix=key,
            )


def plot_batch_size_throughput_gpu_util(
    df: pd.DataFrame,
    title: str = "Batch size vs Throughput & GPU Utilization",
    output_dir: str = "./plot_out",
    output_filename: str = "batch_size_throughput_gpu_util.png",
    gpus_filter: int | None = None,
):
    """
    For each model, plot a figure with batch size on the x-axis,
    throughput (tiles/s) on the left y-axis, and GPU utilization (%)
    on the right y-axis. Two curves per subplot: one for throughput
    and one for GPU utilization.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: model, batch_size, throughput_mean, gpu_util_percent_mean.
        Optionally 'gpus' to filter by GPU count.
    title : str
        Super-title for the figure.
    output_dir : str
        Directory to save the figure.
    output_filename : str
        File name for the saved figure.
    gpus_filter : int or None
        If set, only rows with this many GPUs are plotted.
    """
    df = df.copy()

    # Clean model names (strip framework prefixes)
    df["model"] = (
        df["model"]
        .str.replace("triton", "", regex=False)
        .str.replace("pytorch", "", regex=False)
        .str.replace("inferenceonly", "", regex=False)
    )

    for col in ["batch_size", "throughput_mean", "gpu_util_percent_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["batch_size", "throughput_mean", "gpu_util_percent_mean"])

    if gpus_filter is not None:
        df["gpus"] = pd.to_numeric(df["gpus"], errors="coerce")
        df = df[df["gpus"] == gpus_filter]

    if df.empty:
        logging.warning("No data to plot for batch-size throughput/GPU-util figure.")
        return

    # Aggregate per (model, batch_size)
    agg = (
        df.groupby(["model", "batch_size"], as_index=False)
        .agg(
            throughput_mean=("throughput_mean", "mean"),
            gpu_util_percent_mean=("gpu_util_percent_mean", "mean"),
        )
        .sort_values(["model", "batch_size"])
    )

    models = sorted(agg["model"].unique())
    # move the last entry to the second position
    models = [models[0], models[-1]] + models[1:-1]
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    color_throughput = "#1f77b4"
    color_gpu_util = "#ff7f0e"

    # Compute shared y-limits across all models
    throughput_max = agg["throughput_mean"].max()
    throughput_ylim = (0, throughput_max * 1.1)
    gpu_util_ylim = (0, 105)

    all_ax_right = []

    for idx, model in enumerate(models):
        ax_left = axes[0, idx]
        model_data = agg[agg["model"] == model].sort_values("batch_size")

        batch_sizes = model_data["batch_size"].values
        throughputs = model_data["throughput_mean"].values
        gpu_utils = model_data["gpu_util_percent_mean"].values

        # Use evenly-spaced integer positions so ticks are equally spaced
        x_positions = list(range(len(batch_sizes)))

        # Left y-axis: throughput
        ax_left.plot(
            x_positions,
            throughputs,
            marker="o",
            color=color_throughput,
            linewidth=2,
        )
        ax_left.set_xlabel("Batch Size", fontsize=13)
        ax_left.set_xticks(x_positions)
        ax_left.set_xticklabels([str(int(bs)) for bs in batch_sizes])
        ax_left.set_title(model, fontsize=14, fontweight="bold")
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim(throughput_ylim)

        # Only show left y-axis label and tick labels on the first subplot
        if idx == 0:
            ax_left.set_ylabel("tiles / s", fontsize=13, color=color_throughput)
            ax_left.tick_params(axis="y", labelcolor=color_throughput)
        else:
            ax_left.set_ylabel("")
            ax_left.tick_params(axis="y", labelleft=False)

        # Right y-axis: GPU utilization
        ax_right = ax_left.twinx()
        all_ax_right.append(ax_right)
        ax_right.plot(
            x_positions,
            gpu_utils,
            marker="s",
            color=color_gpu_util,
            linewidth=2,
            linestyle="--",
        )
        ax_right.set_ylim(gpu_util_ylim)

        # Only show right y-axis label and tick labels on the last subplot
        if idx == n_models - 1:
            ax_right.set_ylabel(
                "GPU Utilization (%)", fontsize=13, color=color_gpu_util
            )
            ax_right.tick_params(axis="y", labelcolor=color_gpu_util)
        else:
            ax_right.set_ylabel("")
            ax_right.tick_params(axis="y", labelright=False)

    # Shared legend at the bottom of the figure
    handle_throughput = mlines.Line2D(
        [],
        [],
        color=color_throughput,
        marker="o",
        linewidth=2,
        label="tiles/s",
    )
    handle_gpu_util = mlines.Line2D(
        [],
        [],
        color=color_gpu_util,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="GPU util %",
    )
    fig.legend(
        handles=[handle_throughput, handle_gpu_util],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=12,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logging.info("Saved plot to %s", plot_path)
    plt.close()


STANDARD_BATCH_SIZES = sorted([32, 64, 128, 256])
ALLOWED_GPU_TICKS = [1, 2, 4, 6, 8]


def plot_gpu_scaling_throughput_by_batch_size(
    df: pd.DataFrame,
    title: str = "GPU scaling — Throughput by batch size",
    output_dir: str = "./plot_out",
    output_filename: str = "gpu_scaling_throughput_by_batchsize.png",
):
    """
    One subplot per model.
    X-axis: number of GPUs (evenly spaced ticks for 1, 2, 4, 6, 8).
    Left y-axis (shared): throughput (tiles/s) — solid lines, one per batch size.
    Right y-axis (shared): GPU utilization (%) — dashed lines, same color per batch size.
    A single shared legend at the bottom.
    """
    import seaborn as sns

    df = df.copy()

    # Clean model names
    df["model"] = (
        df["model"]
        .str.replace("triton", "", regex=False)
        .str.replace("pytorch", "", regex=False)
        .str.replace("inferenceonly", "", regex=False)
    )

    for col in ["gpus", "batch_size", "throughput_mean", "gpu_util_percent_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["gpus", "batch_size", "throughput_mean"])

    # Aggregate per (model, gpus, batch_size)
    agg_cols = {"throughput_mean": ("throughput_mean", "mean")}
    if df["gpu_util_percent_mean"].notna().any():
        agg_cols["gpu_util_percent_mean"] = ("gpu_util_percent_mean", "mean")
    agg = (
        df.groupby(["model", "gpus", "batch_size"], as_index=False)
        .agg(**agg_cols)
        .sort_values(["model", "batch_size", "gpus"])
    )
    has_gpu_util = "gpu_util_percent_mean" in agg.columns

    available_batch_sizes = sorted(
        [bs for bs in STANDARD_BATCH_SIZES if bs in agg["batch_size"].unique()]
    )
    if not available_batch_sizes:
        logging.warning("No standard batch sizes found for GPU-scaling plot.")
        return

    models = sorted(agg["model"].unique())
    n_models = len(models)
    if n_models == 0:
        logging.warning("No models found for GPU-scaling plot.")
        return

    # Shared y-limits
    y_max = agg["throughput_mean"].max() * 1.1
    gpu_util_ylim = (0, 105)

    # Evenly-spaced x positions for GPU ticks
    gpu_x = {gpu: i for i, gpu in enumerate(ALLOWED_GPU_TICKS)}
    x_positions = list(range(len(ALLOWED_GPU_TICKS)))

    # Colors & markers per batch size
    palette = sns.color_palette("colorblind", n_colors=len(available_batch_sizes))
    bs_colors = dict(zip(available_batch_sizes, palette))
    markers_solid = ["o", "s", "D", "^"]
    markers_open = ["v", "P", "X", "*"]
    bs_markers_throughput = dict(
        zip(available_batch_sizes, markers_solid[: len(available_batch_sizes)])
    )
    bs_markers_gpu_util = dict(
        zip(available_batch_sizes, markers_open[: len(available_batch_sizes)])
    )

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax_left = axes[0, idx]
        model_data = agg[agg["model"] == model]

        # --- Left y-axis: throughput (solid lines) ---
        for bs in available_batch_sizes:
            bs_data = model_data[model_data["batch_size"] == bs].sort_values("gpus")
            xs = [gpu_x[g] for g in bs_data["gpus"].values if g in gpu_x]
            ys = bs_data.loc[
                bs_data["gpus"].isin(gpu_x.keys()), "throughput_mean"
            ].values
            ax_left.plot(
                xs,
                ys,
                marker=bs_markers_throughput[bs],
                color=bs_colors[bs],
                linewidth=2,
            )

        ax_left.set_xlabel("GPUs", fontsize=13)
        ax_left.set_xticks(x_positions)
        ax_left.set_xticklabels([str(g) for g in ALLOWED_GPU_TICKS], fontsize=12)
        ax_left.set_title(model, fontsize=14, fontweight="bold")
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim(0, y_max)

        # Only show left y-axis label/ticks on the leftmost subplot
        if idx == 0:
            ax_left.set_ylabel("tiles / s", fontsize=13)
        else:
            ax_left.set_ylabel("")
            ax_left.tick_params(axis="y", labelleft=False)

        # --- Right y-axis: GPU utilization (dashed lines) ---
        if has_gpu_util:
            ax_right = ax_left.twinx()
            for bs in available_batch_sizes:
                bs_data = model_data[model_data["batch_size"] == bs].sort_values("gpus")
                xs = [gpu_x[g] for g in bs_data["gpus"].values if g in gpu_x]
                ys = bs_data.loc[
                    bs_data["gpus"].isin(gpu_x.keys()), "gpu_util_percent_mean"
                ].values
                ax_right.plot(
                    xs,
                    ys,
                    marker=bs_markers_gpu_util[bs],
                    color=bs_colors[bs],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )
            ax_right.set_ylim(gpu_util_ylim)

            # Only show right y-axis label/ticks on the rightmost subplot
            if idx == n_models - 1:
                ax_right.set_ylabel("GPU Utilization (%)", fontsize=13)
            else:
                ax_right.set_ylabel("")
                ax_right.tick_params(axis="y", labelright=False)

    # --- Single shared legend at the bottom ---
    handles = []
    for bs in available_batch_sizes:
        # Throughput handle (solid)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=bs_colors[bs],
                marker=bs_markers_throughput[bs],
                linewidth=2,
                label=f"bs={int(bs)} tiles/s",
            )
        )
    if has_gpu_util:
        for bs in available_batch_sizes:
            # GPU util handle (dashed)
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=bs_colors[bs],
                    marker=bs_markers_gpu_util[bs],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                    label=f"bs={int(bs)} GPU util %",
                )
            )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(available_batch_sizes),
        frameon=True,
        fontsize=11,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logging.info("Saved plot to %s", plot_path)
    plt.close()


def main():
    args = parse_args()
    read_and_write_data(args.tensorboard_dir, args.output_dir)

    # --- Generate batch-size vs throughput & GPU utilization plots ---
    for run_prefix in ["triton"]:
        for key_label, key in [("", ""), ("inferenceonly", "inferenceonly")]:
            csv_path = os.path.join(
                args.output_dir, f"{run_prefix}{key}{GPU_SCALING_CSV}"
            )
            if not os.path.exists(csv_path):
                logging.info("CSV not found, skipping: %s", csv_path)
                continue

            df = pd.read_csv(csv_path)
            suffix_label = (
                " (inference only)" if key == "inferenceonly" else " (with I/O)"
            )

            # If multiple GPU counts exist, plot one figure per GPU count
            # if "gpus" in df.columns:
            #     for gpus in sorted(df["gpus"].unique()):
            #         plot_batch_size_throughput_gpu_util(
            #             df,
            #             title=f"{run_prefix.capitalize()}{suffix_label} — {int(gpus)} GPU(s)",
            #             output_dir=args.output_dir,
            #             output_filename=f"{run_prefix}{key}_batch_throughput_gpuutil_{int(gpus)}gpu.png",
            #             gpus_filter=int(gpus),
            #         )
            # else:
            #     plot_batch_size_throughput_gpu_util(
            #         df,
            #         title=f"{run_prefix.capitalize()}{suffix_label}",
            #         output_dir=args.output_dir,
            #         output_filename=f"{run_prefix}{key}_batch_throughput_gpuutil.png",
            #     )

    # --- Generate GPU-scaling plots ---
    for run_prefix in ["triton"]:
        for key_label, key in [("", ""), ("inferenceonly", "inferenceonly")]:
            csv_path = os.path.join(
                args.output_dir, f"{run_prefix}{key}{GPU_SCALING_CSV}"
            )
            if not os.path.exists(csv_path):
                logging.info("CSV not found, skipping: %s", csv_path)
                continue

            df = pd.read_csv(csv_path)
            suffix_label = (
                " (inference only)" if key == "inferenceonly" else " (with I/O)"
            )
            plot_gpu_scaling_throughput_by_batch_size(
                df,
                title=f"{run_prefix.capitalize()}{suffix_label}",
                output_dir=args.output_dir,
                output_filename=f"{run_prefix}{key}_gpu_scaling.png",
            )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
