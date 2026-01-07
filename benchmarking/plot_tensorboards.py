#!/usr/bin/env python
import glob

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="Plot tensorboard data")
# I usually get the tensorboard directories using a bash-command like:
# for serv in sn1 gpu2 gpu3 gpu4 gpu5 gpu6; do scp -r $serv:simple_triton/tensorboard_out tb_$serv; done
# then you can run this with python plot_tensorboard.py --tensorboard-dir-glob "tb_gpu*"
parser.add_argument(
    "--tensorboard-dir-glob",
    type=str,
    help="Glob pattern for tensorboard directories",
    required=False,
    default="tb_gpu*",
)


def extract_tensorboard_data(logdir):
    scalar_data = {}

    for event in tf.compat.v1.train.summary_iterator(logdir):
        for value in event.summary.value:
            if value.tag not in scalar_data:
                scalar_data[value.tag] = []
            scalar_data[value.tag].append((event.step, value.simple_value))

    dataframes = {}
    for tag, values in scalar_data.items():
        if values:
            steps, scalars = zip(*values)
            dataframes[tag] = pd.DataFrame({"step": steps, "value": scalars})

    return dataframes


def plot_tensorboard_data(dataframes, named_run, keys):
    for k in keys:
        df = dataframes[k]
        # only keep 98th percentile of df
        quantile = 0.98
        df = df[df["value"] < df["value"].quantile(quantile)]
        plt.figure(figsize=(8, 6))
        plt.plot(df["step"], df["value"], label=k)
        plt.xlabel("Step")
        plt.ylabel(k)
        plt.title(f"{named_run}: {quantile}%quantile {k} over steps")
        # add average
        avg = df["value"].mean()
        plt.axhline(avg, color="r", linestyle="--", label=f"Average: {avg:.2f}")
        # add running average with step window
        window = 50
        running_avg = df["value"].rolling(window=window).mean()
        plt.plot(df["step"], running_avg, label=f"Running average (window={window})")
        plt.legend()
        plt.show()


def main(tensorboard_dir_glob):
    for dir in glob.glob("tb_sn1*"):
        for named_run in glob.glob(f"{dir}/*"):
            for file in os.listdir(named_run):
                event_file = os.path.join(named_run, file)
                try:
                    keys = [
                        "chars_read_per_s",
                        # "retrieval_total_avg",
                        "time_elapsed",
                    ]
                    dataframes = extract_tensorboard_data(event_file)
                    plot_tensorboard_data(dataframes, named_run, keys)
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")
                    continue


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.tensorboard_dir_glob)
