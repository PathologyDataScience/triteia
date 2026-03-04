"""
Get pure WSI read speed.
This script will iterate over a given directory or WSI file and time the tile throughput per second.
The tile iteration is done in threads to simulate the main program.

To get pure theoretical throughput on a given disk, run fio with the following command (experiment with "--bs" and "--iodepth")":
fio --name=readmax --filename=./test_data/wsi/TCGA-AN-A0G0-01Z-00-DX1.svs --rw=read --bs=512KB --ioengine=io_uring --iodepth=32 --numjobs=1     --direct=1 --time_based --runtime=30 --group_reportingI
theoretical_throughput_max_speed_mb = 3000
theoretical_throughput_max_speed_char = theoretical_throughput_max_speed_mb * 1024 * 1024
writer.add_text("char_read_theoretical_max_speed", str(theoretical_throughput_max_speed_char), 0)

to get estimated number of raw disk content for a given .svs:
(adjust the 'sed 1d;3d' and filename)
 tiffdump -m 100000  ~/simple_triton/test_data/wsi/TCGA-AN-A0G0-01Z-00-DX1.svs | grep TileByteCounts | sed '1d;3d' | awk '{ for (i=6; i<=NF; i++) sum += $i } END { print sum }'
"""

import csv
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ensure we're loading the simple_triton in this directory, not installed in path
from sys import path
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

path.append(os.path.join(os.path.dirname(__file__), "../simple_triton"))
from simple_triton.feature_extraction import study
from simple_triton.tile_iterators import TiffPrefetch

from util import parse_args, clear_cache


def main():
    args = parse_args()

    run_times = []
    run_metrics = []  # Store additional metrics

    clear_cache()

    for f in args.wsi_path:
        print(f"Processing {f}")
        hs_study = study(
            f,
            t=(args.tile_size, args.tile_size),
            chunk=(args.chunk_size, args.chunk_size) if args.chunk_size >= 0 else None,
            objective=args.mag,
        )
        iterator = TiffPrefetch(
            hs_study,
            dtype=np.uint8,
            nchw=args.nchw,
            icc=args.icc,
            batch=args.batch_size,
            prefetch=args.prefetch,
            workers=args.workers,
        )
        number_of_tiles = sum(
            len(iterator.read_kwargs[i]) for i in range(len(iterator.read_kwargs))
        )

        def minimal_work(b):
            return b[0].view().nbytes

        number_of_batches = math.ceil(number_of_tiles / iterator.batch) - 1
        with ThreadPoolExecutor(8) as executor:
            start_time = perf_counter()
            _ = sum(executor.map(minimal_work, iterator))
            elapsed_time = perf_counter() - start_time
        # print(f"{f}: {num_bytes} bytes")

        # Calculate rates
        tiles_per_second = number_of_tiles / elapsed_time
        batches_per_second = number_of_batches / elapsed_time

        run_times.append((f, elapsed_time))
        run_metrics.append(
            (
                f,
                elapsed_time,
                number_of_tiles,
                number_of_batches,
                tiles_per_second,
                batches_per_second,
            )
        )

    times = list(map(lambda x: x[1], run_times))
    # Compute statistics
    avg_time = np.mean(times)
    median_time = np.median(times)
    best_time = np.min(times)
    worst_time = np.max(times)

    print("\n" + "=" * 50)
    print("Running Time Statistics:")
    print("=" * 50)
    print(f"Best time:    {best_time:.2f} seconds")
    print(f"Worst time:   {worst_time:.2f} seconds")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Median time:  {median_time:.2f} seconds")
    print("=" * 50)

    # ---------------------------------------------------------------------
    # Save enhanced metrics to CSV
    # ---------------------------------------------------------------------
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    csv_filename = os.path.join(
        mydir,
        f"wsi_read_speed_b{args.batch_size}_w{args.workers}_p{args.prefetch}_c{args.chunk_size}.csv",
    )
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(
            [
                "run_index",
                "filename",
                "time_seconds",
                "number_of_tiles",
                "number_of_batches",
                "tiles_per_second",
                "batches_per_second",
            ]
        )
        # Per-run values
        for idx, (f, t, tiles, batches, tiles_per_sec, batches_per_sec) in enumerate(
            run_metrics, start=1
        ):
            writer.writerow(
                [
                    idx,
                    f,
                    t,
                    tiles,
                    batches,
                    f"{tiles_per_sec:.2f}",
                    f"{batches_per_sec:.2f}",
                ]
            )

    print(f"Run times CSV saved as: {csv_filename}")

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual run times
    run_indices = list(range(1, len(times) + 1))
    ax.plot(
        run_indices,
        times,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="Run Time",
        color="steelblue",
    )

    # Plot average line
    ax.axhline(
        y=avg_time,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_time:.2f}s)",
    )

    # Plot median line
    ax.axhline(
        y=median_time,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median ({median_time:.2f}s)",
    )

    # Plot best and worst
    ax.axhline(
        y=best_time,
        color="lightgreen",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Best ({best_time:.2f}s)",
    )
    ax.axhline(
        y=worst_time,
        color="lightcoral",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Worst ({worst_time:.2f}s)",
    )

    # Formatting
    ax.set_xlabel("Run Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("WSI Read Speed - Running Times", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(run_indices)

    # Add some padding to y-axis
    y_range = worst_time - best_time
    ax.set_ylim(
        best_time - 0.1 * y_range if y_range > 0 else best_time - 1,
        worst_time + 0.1 * y_range if y_range > 0 else worst_time + 1,
    )

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    plot_filename = os.path.join(
        mydir,
        f"wsi_read_speed_b{args.batch_size}_w{args.workers}_p{args.prefetch}_c{args.chunk_size}.png",
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as: {plot_filename}")

    plt.show()

    # =========================================================================
    # Test 2: Time to First Batch (averaged over 3 runs)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Time to First Batch Test (3 runs)")
    print("=" * 50)

    first_batch_times = []
    clear_cache()

    for f in args.wsi_path:
        hs_study = study(
            f,
            t=(args.tile_size, args.tile_size),
            chunk=(args.chunk_size, args.chunk_size) if args.chunk_size >= 0 else None,
            objective=args.mag,
        )
        iterator = TiffPrefetch(
            hs_study,
            dtype=np.uint8,
            nchw=args.nchw,
            icc=args.icc,
            batch=args.batch_size,
            prefetch=args.prefetch,
            workers=args.workers,
        )
        start_time = perf_counter()
        # Get only the first batch
        first_batch = next(iter(iterator))
        first_batch_time = perf_counter() - start_time
        first_batch_times.append((f, first_batch_time))
        print(f"  Time to first batch: {first_batch_time:.4f} seconds")

    times = list(map(lambda x: x[1], first_batch_times))
    # Compute statistics for first batch times
    avg_first_batch = np.mean(times)
    median_first_batch = np.median(times)
    best_first_batch = np.min(times)
    worst_first_batch = np.max(times)

    print("\n" + "-" * 50)
    print("First Batch Statistics:")
    print("-" * 50)
    print(f"Best time:    {best_first_batch:.4f} seconds")
    print(f"Worst time:   {worst_first_batch:.4f} seconds")
    print(f"Average time: {avg_first_batch:.4f} seconds")
    print(f"Median time:  {median_first_batch:.4f} seconds")
    print("=" * 50)

    # ---------------------------------------------------------------------
    # Save only per-run first batch times to CSV (no summary statistics rows)
    # ---------------------------------------------------------------------
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    first_batch_csv_filename = os.path.join(
        mydir,
        f"wsi_first_batch_b{args.batch_size}_w{args.workers}_p{args.prefetch}_c{args.chunk_size}.csv",
    )
    with open(first_batch_csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["run_index", "filename", "time_seconds"])
        # Per-run values
        for idx, (f, t) in enumerate(first_batch_times, start=1):
            writer.writerow([idx, f, t])

    print(f"First batch times CSV saved as: {first_batch_csv_filename}")

    # Create line plot for first batch times
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot individual run times
    run_indices = list(range(1, len(first_batch_times) + 1))
    ax2.plot(
        run_indices,
        times,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=10,
        label="Time to First Batch",
        color="steelblue",
        markeredgecolor="black",
        markeredgewidth=1.5,
    )

    # Add value labels on each point
    for x, y in zip(run_indices, times):
        ax2.text(
            x,
            (
                y + (worst_first_batch - best_first_batch) * 0.05
                if worst_first_batch > best_first_batch
                else y + 0.01
            ),
            f"{y:.4f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot average line
    ax2.axhline(
        y=avg_first_batch,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_first_batch:.4f}s)",
    )

    # Plot median line
    ax2.axhline(
        y=median_first_batch,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median ({median_first_batch:.4f}s)",
    )

    # Plot best and worst
    ax2.axhline(
        y=best_first_batch,
        color="lightgreen",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Best ({best_first_batch:.4f}s)",
    )
    ax2.axhline(
        y=worst_first_batch,
        color="lightcoral",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Worst ({worst_first_batch:.4f}s)",
    )

    # Formatting
    ax2.set_xlabel("Run Number", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title("Time to First Batch - Comparison", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xticks(run_indices)

    # Add some padding to y-axis
    y_range = worst_first_batch - best_first_batch
    ax2.set_ylim(
        best_first_batch - 0.1 * y_range if y_range > 0 else best_first_batch - 0.01,
        worst_first_batch + 0.2 * y_range if y_range > 0 else worst_first_batch + 0.02,
    )

    plt.tight_layout()

    # Save the plot
    plot_filename2 = os.path.join(
        mydir,
        f"wsi_first_batch_b{args.batch_size}_w{args.workers}_p{args.prefetch}_c{args.chunk_size}.png",
    )
    plt.savefig(plot_filename2, dpi=300, bbox_inches="tight")
    print(f"\nFirst batch plot saved as: {plot_filename2}")

    plt.show()


if __name__ == "__main__":
    main()
    print("Done!")
    import os

    os._exit(0)
