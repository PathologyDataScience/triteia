#!/usr/bin/env python3

"""
Plot read speed benchmark results
the files are in a directory given by args. In directory there are files named
"wsi_read_speed_<batch_size>_w<workers>_p<prefetch>.csv"

a CSV file can look like this:
run_index,filename,time_seconds,number_of_tiles,number_of_batches,tiles_per_second,batches_per_second
1,/data/5/TCGA-5L-AAT0-01Z-00-DX1.5E171263-30BF-4C6B-88A1-E8EA0522A861.svs,6.9609404880029615,18221,71,2617.61,10.20
2,/data/5/TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs,13.663311332013109,49288,192,3607.32,14.05
3,/data/5/TCGA-A2-A04Y-01Z-00-DX1.4DC97AD6-4806-4A3A-A998-FD36F93590A4.svs,6.795668399994611,30636,119,4508.17,17.51

this file first finds the top 2 optimal configurations per batch size, then prints it
then, for a given worker and prefetch setting, it prints the maximum and average "tiles_per_second" for each batch size
"""
import argparse
import glob
import logging
import os
import re
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tensorboard data")
    parser.add_argument(
        "--input-path",
        type=str,
        help="input directory for csv file",
        default="/terrahome/backup-2feb/read_speeds",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory for csv file",
        default="./plot_out",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input directory {args.input_path} not found.")

    return args


def main():
    args = parse_args()

    files = glob.glob(args.input_path + os.path.sep + "wsi_read_speed*.csv")
    assert len(files) > 0, "No csv files found in input directory"

    parse_re = re.compile(
        rf"wsi_read_speed_b(?P<batch>[0-9]+)_w(?P<workers>[0-9]+)_p(?P<prefetch>[0-9]+)_c(?P<chunk>[0-9]+).csv"
    )
    df_li = []
    for f in files:
        bn = os.path.basename(f)
        m = parse_re.match(bn)
        if not m:
            raise ValueError(f"Could not parse filename {bn} with {parse_re.pattern}")

        df = pd.read_csv(f)
        batch = int(m["batch"])
        workers = int(m["workers"])
        prefetch = int(m["prefetch"])
        chunk = int(m["chunk"])

        data = {
            "batch": batch,
            "workers": workers,
            "prefetch": prefetch,
            "chunk": chunk,
            "max": int(df["tiles_per_second"].max()),
            "avg": int(df["tiles_per_second"].mean()),
            "origin": bn,
        }
        df_li.append(pd.DataFrame([data]))

    df = pd.concat(df_li)

    top2_configs_max = (
        df.groupby("batch").apply(lambda x: x.nlargest(1, "max")).reset_index(drop=True)
    )
    print("Best configuration for MAX read speed")
    print(top2_configs_max.to_string(index=False))

    print("Best configuration for MEAN/AVG read speed")
    top2_configs_avg = (
        df.groupby("batch").apply(lambda x: x.nlargest(1, "avg")).reset_index(drop=True)
    )
    print(top2_configs_avg.to_string(index=False))
    top2_configs_avg.to_csv(
        os.path.join(args.output_dir, "read_config_speeds.csv"), index=False
    )

    best_chunk = 896
    best_workers = 64
    best_prefetch = 16
    # get row with best config
    best_config_row = df[
        (df["chunk"] == best_chunk)
        & (df["workers"] == best_workers)
        & (df["prefetch"] == best_prefetch)
    ]
    # sort by batch size
    best_config_row = best_config_row.sort_values(by="batch", ascending=True)
    if best_config_row.empty:
        raise ValueError(
            f"Best config not found: chunk={best_chunk}, workers={best_workers}, prefetch={best_prefetch}"
        )
    print(f"Best config assumption: {best_config_row.to_string(index=False)}")

    first_128 = glob.glob(
        args.input_path
        + os.path.sep
        + f"wsi_first_batch_b128_w{best_workers}_p{best_prefetch}_c{best_chunk}.csv"
    )[0]
    assert os.path.exists(first_128), f"First batch plot not found: {first_128}"
    first_256 = glob.glob(
        args.input_path
        + os.path.sep
        + f"wsi_first_batch_b256_w{best_workers}_p{best_prefetch}_c{best_chunk}.csv"
    )[0]
    assert os.path.exists(first_256), f"First batch plot not found: {first_256}"
    first_512 = glob.glob(
        args.input_path
        + os.path.sep
        + f"wsi_first_batch_b512_w{best_workers}_p{best_prefetch}_c{best_chunk}.csv"
    )[0]
    assert os.path.exists(first_512), f"First batch plot not found: {first_512}"

    # Collect results for CSV output
    results = []
    for file in [first_128, first_256, first_512]:
        df = pd.read_csv(file)
        m = df["time_seconds"].mean()
        results.append({"filename": os.path.basename(file), "average_time_seconds": m})

    # Write results to CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, "read_first_batch.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
    logging.info("Done")
