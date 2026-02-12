#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorBoard power usage summarizer (fresh CSV schema).

Reads scalar tags from TensorBoard event files, writes a normalized CSV, and prints
summary stats grouped by GPU count extracted from the run path (e.g., ".../gigapathgpu4bs64").

Default tags:
  GPU: energy_usage_gpu_watt_total
  CPU: energy_usage_cpu_watt_total
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

GPU_RE = re.compile(r"gpu(?P<gpus>\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Tags:
    gpu: str
    cpu: str
    time: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize TensorBoard power usage by GPU config"
    )
    p.add_argument(
        "--tensorboard-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing TensorBoard runs (or parent directories of runs).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="./plot_out/power-usage.csv",
        help="Output CSV path to write (overwritten).",
    )
    p.add_argument(
        "--gpu-tag",
        type=str,
        default="energy_usage_gpu_watt_total",
        help="TensorBoard scalar tag for GPU power.",
    )
    p.add_argument(
        "--cpu-tag",
        type=str,
        default="energy_usage_cpu_watt_total",
        help="TensorBoard scalar tag for CPU power.",
    )
    p.add_argument(
        "--allowed-substrings",
        type=str,
        nargs="*",
        default=["uni", "gigapath", "resnet50"],
        help="Only include runs whose path contains any of these substrings (case-insensitive). "
        "Pass an empty list to include all runs.",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable allowed-substrings filtering (include all runs).",
    )
    p.add_argument(
        "--scale-gpu",
        type=float,
        default=1,
        help="Divide GPU watt by this factor (useful if your logged units need scaling).",
    )
    p.add_argument(
        "--scale-cpu",
        type=float,
        default=1,
        help="Divide CPU watt by this factor (useful if your logged units need scaling).",
    )
    return p.parse_args()


def _candidate_run_dirs(roots: Iterable[str]) -> list[str]:
    out: set[str] = set()
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise ValueError(f"Not a directory: {root}")
        out.add(root)

        # Include direct children as likely run dirs
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                out.add(p)
    return sorted(out)


def _extract_gpus_from_path(path: str) -> Optional[int]:
    m = GPU_RE.search(path)
    if not m:
        return None
    try:
        return int(m.group("gpus"))
    except Exception:
        return None


def _infer_model_from_path(path: str, allowed_substrings: list[str]) -> str:
    low = path.lower()
    for s in allowed_substrings:
        if s.lower() in low:
            return s.lower()
    return "unknown"


def extract_scalar_values(logdir: str, tag: str) -> list[float]:
    """
    Extract scalar values for `tag` from all events.out.tfevents.* files under logdir.
    """
    # Lazy import: only needed if we actually parse event files.
    import tensorflow as tf

    event_files = glob.glob(
        os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True
    )
    if not event_files:
        return []

    values: list[float] = []
    for ef in sorted(event_files):
        try:
            for e in tf.compat.v1.train.summary_iterator(ef):
                if not getattr(e, "summary", None):
                    continue
                for v in e.summary.value:
                    if v.tag != tag:
                        continue
                    if hasattr(v, "simple_value"):
                        try:
                            values.append(float(v.simple_value))
                        except Exception:
                            continue
        except Exception:
            # Corrupt/partial event file; skip safely.
            continue
    return values


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_power_usage_dataframe(
    roots: list[str],
    tags: Tags,
    allowed_substrings: list[str],
    disable_filter: bool,
    scale_gpu: float,
    scale_cpu: float,
) -> pd.DataFrame:
    run_dirs = _candidate_run_dirs(roots)
    rows: list[dict] = []

    for d in run_dirs:
        low = d.lower()
        if not disable_filter and allowed_substrings:
            if not any(s.lower() in low for s in allowed_substrings):
                continue

        gpu_values = extract_scalar_values(d, tags.gpu)
        cpu_values = extract_scalar_values(d, tags.cpu)
        time_values = extract_scalar_values(d, tags.time)

        # Skip dirs that don't have either tag
        if not gpu_values and not cpu_values:
            continue

        # remove any value less than or equal to zero: the counter may be have been reset
        gpu_values = [v for v in gpu_values if v > 0]
        cpu_values = [v for v in cpu_values if v > 0]

        mean_gpu = np.mean(gpu_values) / scale_gpu
        mean_cpu = np.mean(cpu_values) / scale_cpu
        mean_time = np.mean(time_values)

        mean_total = mean_gpu + mean_cpu

        gpus = _extract_gpus_from_path(d)

        rows.append(
            {
                "model": _infer_model_from_path(d, allowed_substrings),
                "run": os.path.basename(os.path.normpath(d)),
                "logdir": d,
                "gpus": gpus,
                "mean_gpu_watt": mean_gpu,
                "mean_cpu_watt": mean_cpu,
                "mean_total_watt": mean_total,
                "total_watt": sum(gpu_values + cpu_values),
                "total_watt_cpu": sum(cpu_values),
                "total_watt_gpu": sum(gpu_values),
                "total_time": np.sum(time_values),
                "mean_time": mean_time,
                "n_gpu_points": int(len(gpu_values)),
                "n_cpu_points": int(len(cpu_values)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize dtypes
    df["gpus"] = pd.to_numeric(df["gpus"], errors="coerce").astype("Int64")
    for c in (
        "mean_gpu_watt",
        "mean_cpu_watt",
        "mean_total_watt",
        "total_watt",
        "total_watt_cpu",
        "total_watt_gpu",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("n_gpu_points", "n_cpu_points"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["model", "run", "logdir"]).reset_index(drop=True)
    return df


def summarize_by_gpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary per GPU configuration for each watt column:
      - mean / median / min / max / std for: total, cpu, gpu
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "gpus",
                "count_runs",
                "total_mean_watt",
                "total_median_watt",
                "total_min_watt",
                "total_max_watt",
                "total_std_watt",
                "cpu_mean_watt",
                "cpu_median_watt",
                "cpu_min_watt",
                "cpu_max_watt",
                "cpu_std_watt",
                "gpu_mean_watt",
                "gpu_median_watt",
                "gpu_min_watt",
                "gpu_max_watt",
                "gpu_std_watt",
                "total_watt",
                "total_watt_cpu",
                "total_watt_gpu",
                "total_time",
                "mean_time",
            ]
        )

    required = {
        "gpus",
        "mean_total_watt",
        "mean_cpu_watt",
        "mean_gpu_watt",
        "total_watt",
        "total_watt_cpu",
        "total_watt_gpu",
        "total_time",
        "mean_time",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    tmp = df.dropna(subset=["gpus"]).copy()
    tmp["gpus"] = tmp["gpus"].astype(int)

    # Count all runs per GPU config, even if some watt columns are NaN for some runs.
    count_runs = tmp.groupby("gpus", as_index=False).agg(count_runs=("logdir", "count"))

    watt_cols = {
        "total": "mean_total_watt",
        "cpu": "mean_cpu_watt",
        "gpu": "mean_gpu_watt",
    }
    stats = ["mean", "median", "min", "max", "std"]

    agg = tmp.groupby("gpus")[list(watt_cols.values())].agg(stats).reset_index()

    # Flatten MultiIndex columns: ('mean_total_watt','mean') -> 'total_mean_watt'
    rename_map: dict[str, str] = {}
    for prefix, col in watt_cols.items():
        for stat in stats:
            rename_map[f"{col}_{stat}"] = f"{prefix}_{stat}_watt"

    flat_cols = ["gpus"]
    for col in list(watt_cols.values()):
        for stat in stats:
            flat_cols.append(f"{col}_{stat}")

    agg.columns = flat_cols
    agg = agg.rename(columns=rename_map)

    # Aggregate total_watt columns
    total_watt_agg = (
        tmp.groupby("gpus")[["total_watt", "total_watt_cpu", "total_watt_gpu"]]
        .sum()
        .reset_index()
    )

    # Aggregate time_elapsed
    time_elapsed_agg = tmp.groupby("gpus", as_index=False).agg(
        time_elapsed_total=("total_time", "sum"),
        time_elapsed_mean=("mean_time", "sum"),
    )

    summary = (
        count_runs.merge(agg, on="gpus", how="inner")
        .merge(total_watt_agg, on="gpus", how="inner")
        .merge(time_elapsed_agg, on="gpus", how="inner")
        .sort_values("gpus")
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Cache behavior: if a CSV already exists, reuse it and do NOT rescan TensorBoards.
    out_csv = os.path.abspath(args.out_csv)
    fallback_candidates = [
        out_csv,
        os.path.abspath("power-usage"),
        os.path.abspath("power-usage.csv"),
    ]
    existing_csv = next((p for p in fallback_candidates if os.path.isfile(p)), None)

    if existing_csv is not None:
        logging.info(
            "Found existing CSV (%s). Skipping TensorBoard scan.", existing_csv
        )
        df = pd.read_csv(existing_csv)
    else:
        tags = Tags(gpu=args.gpu_tag, cpu=args.cpu_tag, time="time_elapsed")

        df = build_power_usage_dataframe(
            roots=args.tensorboard_dir,
            tags=tags,
            allowed_substrings=list(args.allowed_substrings or []),
            disable_filter=bool(args.no_filter),
            scale_gpu=args.scale_gpu,
            scale_cpu=args.scale_cpu,
        )

        if df.empty:
            logging.warning(
                "No runs found with tags (%s, %s). Nothing to write/summary.",
                tags.gpu,
                tags.cpu,
            )
            print(df)
            return

        df.to_csv(out_csv, index=False)
        logging.info("Wrote %d rows to %s", len(df), out_csv)

    summary = summarize_by_gpus(df)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
