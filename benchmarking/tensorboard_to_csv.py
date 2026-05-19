#!/usr/bin/env python
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

import pandas as pd

GPU_SCALING_CSV = "gpu_scaling.csv"
LATENCY_GPU_SCALING_CSV = "gpu_latency_scaling.csv"
PYTORCH_GPU_SCALING_CSV = "pytorch_gpu_scaling.csv"
PYTORCH_LATENCY_GPU_SCALING_CSV = "pytorch_gpu_latency_scaling.csv"
LIMIT_SCALING_CSV = "limit_scaling.csv"
INFERENCEONLYTRITON_GPU_SCALING_CSV = "inferenceonlytriton_gpu_scaling.csv"
INFERENCEONLYPYTORCH_GPU_SCALING_CSV = "inferenceonlypytorch_gpu_scaling.csv"
MULTIUSER_SCALING_CSV = "multiuser_scaling.csv"


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
      model, run, logdir, gpus, batch_size, plot_key, throughput_mean, throughput_last

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
                "No tensorboard run directories matched for model '%s' (prefix=%r, suffix=%r) (regex: %r)",
                spec["name"],
                run_prefix,
                run_suffix,
                spec["run_dir_re"],
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
        ],
    )

    if not out.empty:
        out = out.sort_values(["model", "batch_size", "gpus", "run"]).reset_index(
            drop=True
        )
    return out


def build_limit_scaling_dataframe(tensorboard_dirs, plot_key: str) -> pd.DataFrame:
    """
    Scan tensorboard directories matching:
      "<modelname>gpu[0-9]+gpu[1-8]limit[0-9]+"

    Output columns:
      model, run, logdir, gpus, limit, plot_key, throughput_mean, throughput_last
    """
    model_specs = [
        {
            "name": "Prov-GigaPath",
            "run_dir_re": re.compile(r"tritongigapathgpu[1-9]limit[0-9]+$"),
            "parse_re": re.compile(
                r"tritongigapathgpu(?P<gpus>[1-8])limit(?P<limit>[0-9]+)$"
            ),
        },
        {
            "name": "UNI",
            "run_dir_re": re.compile(r"tritonunigpu[1-9]limit[0-9]+$"),
            "parse_re": re.compile(
                r"tritonunigpu(?P<gpus>[1-8])limit(?P<limit>[0-9]+)$"
            ),
        },
        {
            "name": "ResNet-50",
            "run_dir_re": re.compile(r"tritonresnet50gpu[1-9]limit[0-9]+$"),
            "parse_re": re.compile(
                r"tritonresnet50gpu(?P<gpus>[1-8])limit(?P<limit>[0-9]+)$"
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
                "No limit-scaling tensorboard run directories matched for model '%s'",
                spec["name"],
            )
            continue

        for d in run_dirs:
            base = os.path.basename(os.path.normpath(d))
            m = spec["parse_re"].match(base)
            if not m:
                continue

            gpus = int(m.group("gpus"))
            limit = int(m.group("limit"))

            df = extract_tensorboard_data(d, plot_key=plot_key)
            if df.empty:
                logging.warning("No data for key '%s' in %s", plot_key, d)
                continue

            rows.append(
                {
                    "model": spec["name"],
                    "run": base,
                    "logdir": d,
                    "gpus": gpus,
                    "limit": limit,
                    "plot_key": plot_key,
                    "throughput_mean": float(df["value"].mean()),
                    "throughput_last": float(df["value"].iloc[-1]),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "model",
            "run",
            "logdir",
            "gpus",
            "limit",
            "plot_key",
            "throughput_mean",
            "throughput_last",
        ],
    )

    if not out.empty:
        out = out.sort_values(["model", "gpus", "limit", "run"]).reset_index(drop=True)

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
            f"No matching runs with scalar data found for plot key '{plot_key}' with prefix '{run_prefix}' and suffix '{run_suffix}' in {tensorboard_dirs}"
        )

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)
    return df


def load_or_build_limit_scaling(
    tensorboard_dirs, csv_path: str, plot_key: str
) -> pd.DataFrame:
    """
    If csv exists, load it and skip reading TensorBoards.
    Otherwise, build dataframe from TensorBoards and write csv.
    """
    if os.path.exists(csv_path) and os.path.isfile(csv_path):
        logging.info(
            "Found %s; loading cached limit-scaling data (skipping TensorBoard read).",
            csv_path,
        )
        df = pd.read_csv(csv_path)
        return df

    logging.info(
        "%s not found; reading limit-scaling TensorBoards and creating cache.", csv_path
    )
    df = build_limit_scaling_dataframe(tensorboard_dirs, plot_key=plot_key)
    if df.empty:
        raise ValueError(
            f"No matching limit-scaling runs with scalar data found for plot key '{plot_key} in {tensorboard_dirs}'"
        )

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)
    return df


def build_multiuser_scaling_dataframe(tensorboard_dirs, plot_key: str) -> pd.DataFrame:
    """
    Scan tensorboard directories named:
      "<modelname>multiusergpu[1-8]c[0-9]+n[0-9]+"

    c = concurrency level
    n = thread number

    For each (model, gpus, concurrency), aggregate tile throughput across all n by summing.
    """
    model_specs = [
        {
            "name": "Prov-GigaPath",
            "run_dir_re": re.compile(r"tritongigapathmultiusergpu[1-8]c[0-9]+n[0-9]+$"),
            "parse_re": re.compile(
                r"tritongigapathmultiusergpu(?P<gpus>[1-8])c(?P<c>[0-9]+)n(?P<n>[0-9]+)$"
            ),
        },
        {
            "name": "UNI",
            "run_dir_re": re.compile(r"tritonunimultiusergpu[1-8]c[0-9]+n[0-9]+$"),
            "parse_re": re.compile(
                r"tritonunimultiusergpu(?P<gpus>[1-8])c(?P<c>[0-9]+)n(?P<n>[0-9]+)$"
            ),
        },
        {
            "name": "ResNet-50",
            "run_dir_re": re.compile(r"tritonresnet50multiusergpu[1-8]c[0-9]+n[0-9]+$"),
            "parse_re": re.compile(
                r"tritonresnet50multiusergpu(?P<gpus>[1-8])c(?P<c>[0-9]+)n(?P<n>[0-9]+)$"
            ),
        },
        {
            "name": "ResNet-50-trt",
            "run_dir_re": re.compile(
                rf"tritonresnet50_trt_uint8multiusergpu[1-8]c[0-9]+n[0-9]+$"
            ),
            "parse_re": re.compile(
                rf"tritonresnet50_trt_uint8multiusergpu(?P<gpus>[1-8])c(?P<c>[0-9]+)n(?P<n>[0-9]+)$"
            ),
        },
        {
            "name": f"Prov-GigaPath_trt",
            "run_dir_re": re.compile(
                rf"tritongigapath_trt_uint8multiusergpu[1-8]c[0-9]+n[0-9]+$"
            ),
            "parse_re": re.compile(
                rf"tritongigapath_trt_uint8multiusergpu(?P<gpus>[1-8])c(?P<c>[0-9]+)n(?P<n>[0-9]+)$"
            ),
        },
    ]

    candidate_run_dirs = _candidate_run_dirs(tensorboard_dirs)

    per_thread_rows = []
    for spec in model_specs:
        run_dirs = []
        for d in candidate_run_dirs:
            base = os.path.basename(os.path.normpath(d))
            if spec["run_dir_re"].match(base):
                run_dirs.append(d)

        if not run_dirs:
            logging.warning(
                "No multiuser tensorboard run directories matched for model '%s'",
                spec["name"],
            )
            continue

        for d in run_dirs:
            base = os.path.basename(os.path.normpath(d))
            m = spec["parse_re"].match(base)
            if not m:
                continue

            gpus = int(m.group("gpus"))
            concurrency = int(m.group("c"))
            thread_n = int(m.group("n"))

            df = extract_tensorboard_data(d, plot_key=plot_key)
            if df.empty:
                logging.warning("No data for key '%s' in %s", plot_key, d)
                continue

            per_thread_rows.append(
                {
                    "model": spec["name"],
                    "gpus": gpus,
                    "concurrency": concurrency,
                    "thread_n": thread_n,
                    "plot_key": plot_key,
                    "throughput_mean": float(df["value"].mean()),
                    "throughput_last": float(df["value"].iloc[-1]),
                    "run": base,
                    "logdir": d,
                }
            )

    per_thread = pd.DataFrame(
        per_thread_rows,
        columns=[
            "model",
            "gpus",
            "concurrency",
            "thread_n",
            "plot_key",
            "throughput_mean",
            "throughput_last",
            "run",
            "logdir",
        ],
    )
    if per_thread.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "gpus",
                "concurrency",
                "plot_key",
                "threads",
                "throughput_agg_mean",
                "throughput_agg_last",
            ]
        )

    agg = (
        per_thread.groupby(["model", "gpus", "concurrency", "plot_key"], as_index=False)
        .agg(
            threads=("thread_n", "nunique"),
            throughput_agg_mean=("throughput_mean", "sum"),
            throughput_agg_last=("throughput_last", "sum"),
        )
        .sort_values(["model", "gpus", "concurrency"])
        .reset_index(drop=True)
    )
    return agg


def load_or_build_multiuser_scaling(
    tensorboard_dirs, csv_path: str, plot_key: str
) -> pd.DataFrame:
    """
    If csv exists, load it and skip reading TensorBoards.
    Otherwise, build the multiuser aggregate dataframe from TensorBoards and write csv.

    Writes columns:
      model, gpus, concurrency, plot_key, threads, throughput_agg_mean, throughput_agg_last
    """
    if os.path.exists(csv_path) and os.path.isfile(csv_path):
        logging.info(
            "Found %s; loading cached multiuser-scaling data (skipping TensorBoard read).",
            csv_path,
        )
        return pd.read_csv(csv_path)

    logging.info(
        "%s not found; reading multiuser TensorBoards and creating cache.", csv_path
    )
    df = build_multiuser_scaling_dataframe(tensorboard_dirs, plot_key=plot_key)
    if df.empty:
        raise ValueError(
            f"No matching multiuser runs with scalar data found for plot key '{plot_key}'"
        )

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)
    return df


def read_and_write_data(tensorboard_dirs, output_dir):
    plot_key = "throughput_total_tiles_per_second"

    for run_prefix in ["triton", "pytorch"]:
        for run_suffix in ["", "inferenceonly"]:
            df = load_or_build_gpu_scaling(
                tensorboard_dirs=tensorboard_dirs,
                csv_path=os.path.join(
                    output_dir, f"{run_prefix}{run_suffix}{GPU_SCALING_CSV}"
                ),
                plot_key=plot_key,
                run_prefix=run_prefix,
                run_suffix=run_suffix,
            )
            df = load_or_build_gpu_scaling(
                tensorboard_dirs=tensorboard_dirs,
                csv_path=os.path.join(
                    output_dir, f"{run_prefix}{run_suffix}{LATENCY_GPU_SCALING_CSV}"
                ),
                plot_key="latency_mean_ms",
                run_prefix=run_prefix,
                run_suffix=run_suffix,
            )

    # limit_df = load_or_build_limit_scaling(
    #     tensorboard_dirs=tensorboard_dirs,
    #     csv_path=os.path.join(output_dir, LIMIT_SCALING_CSV),
    #     plot_key=plot_key,
    # )

    # Multiuser aggregate CSV + plot
    multiuser_df = load_or_build_multiuser_scaling(
        tensorboard_dirs=tensorboard_dirs,
        csv_path=os.path.join(output_dir, MULTIUSER_SCALING_CSV),
        plot_key=plot_key,
    )


def main():
    args = parse_args()
    read_and_write_data(args.tensorboard_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
