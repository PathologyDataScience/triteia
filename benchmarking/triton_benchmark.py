"""
This code will do inference on a given folder or WSI file.
The only difference from the main client is that it will instrument and collect stats.

The script will print some stats to console, otherwise, the main idea is that data is written to tensorboards
"""
import glob
import math
import os
from contextlib import ExitStack
from pprint import pprint

# ensure we're loading the triteia in this directory, not installed in path
from sys import path
from time import perf_counter

import numpy as np
import tensorflow as tf
import torch
from tensorboardX import GlobalSummaryWriter

# Make sure we're testing the local triteia
path.append(os.path.join(os.path.dirname(__file__), "../triteia"))
from config import PythonConfig, InstanceGroup
from model import TritonModel
from feature_extraction import study, inference, inference_job, initialize_clients
from tile_iterators import TiffPrefetch
from utils import (
    analyze,
    track_method,
    init_tb_writer,
    write_tritonserver_metrics,
    write_analysis_tb,
)
import pynvml

from util import (
    clear_cache,
    convert_seconds_to_hms,
    parse_args,
    get_energy_reads,
    write_energy_stats,
    mock_metadata,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
assert len(tf.config.list_physical_devices("GPU")) == 0

import time
from datetime import datetime


def sleep_until_next_3rd_minute():
    """
    Sleep until the system clock hits the next 3rd minute (0, 3, 6, 9, etc.).
    Ensures at least 1 minute of sleep. If current time is less than 1 minute
    before the next 3rd minute, sleep until the following 3rd minute.

    Examples:
    - 00:01 -> sleep until 00:03 (2 minutes)
    - 00:02:58 -> sleep until 00:06 (3 minutes 2 seconds)
    """
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # Calculate the next 3rd minute mark
    next_3rd_minute = ((current_minute // 3) + 1) * 3

    # Calculate seconds until the next 3rd minute
    minutes_until_next = next_3rd_minute - current_minute
    seconds_until_next = (minutes_until_next * 60) - current_second

    # If less than 60 seconds remain, skip to the following 3rd minute
    if seconds_until_next < 60:
        next_3rd_minute = ((current_minute // 3) + 2) * 3
        minutes_until_next = next_3rd_minute - current_minute
        seconds_until_next = (minutes_until_next * 60) - current_second

    print(f"Current time: {now.strftime('%H:%M:%S')}")
    print(
        f"Sleeping for {seconds_until_next} seconds until minute {next_3rd_minute % 60}..."
    )

    time.sleep(seconds_until_next)

    print(f"Woke up at: {datetime.now().strftime('%H:%M:%S')}")


def plot_throughput(writer, times, batch_num):
    submitted_times = times["submitted"]
    retrieved_times = times["retrieved"]

    # Calculate latency (received - submitted) for each request
    latencies = [
        (received - submitted) * 1000
        for received, submitted in zip(retrieved_times, submitted_times)
    ]
    for i, l in enumerate(latencies):
        writer.add_scalar("latency_ms", l)
    writer.add_scalar("latency_mean_ms", np.mean(latencies))
    return batch_num + len(latencies)


def main():
    args = parse_args()

    if not args.preload:
        config = PythonConfig(
            args.model_name,
            args.max_batch_size,
            instance_group=InstanceGroup(count=args.instance_group),
        )
        model = TritonModel(args.model_name, args.url)
        model.load(config=config.json())
        assert model.is_loaded()
        config = model.get_config()
        pprint(config)

    # explanation: https://docs.kernel.org/power/powercap/powercap.html
    # energy_uj is "current energy counter in micro joules"
    # i.e. we can measure once before start and once after it's ended
    rapl_sockets = glob.glob(
        "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:*/energy_uj"
    )

    pynvml.nvmlInit()

    time_string = datetime.now().strftime("%y%m%d-%H%M%S")
    tb_name = (
        "triton_" + time_string if not args.tensorboard_name else args.tensorboard_name
    )

    writer: GlobalSummaryWriter = init_tb_writer(
        args.output,
        tb_name,
        [args.wsi_path],
        {
            "icc": args.icc,
            "workers": args.workers,
            "batch": args.batch_size,
            "chunk": args.chunk_size,
            "prefetch": args.prefetch,
            "tile_size": args.tile_size,
            "magnification": args.mag,
            "instance-group": args.instance_group,
            "url": args.url,
            "wsi-path": args.wsi_path,
            "inference_only": args.inference_only,
            "numpy": args.numpy,
            "cuda_available": torch.cuda.is_available(),
        },
    )

    total_time_used = 0
    total_throughput = 0

    if args.numpy:
        if args.nchw:
            data = np.random.randn(
                100, args.batch_size, 3, args.tile_size, args.tile_size
            ).astype(dtype=np.uint8)
        else:
            data = np.random.randn(
                100, args.batch_size, args.tile_size, args.tile_size, 3
            ).astype(dtype=np.uint8)
        data = list(map(lambda l: (l, mock_metadata(batch_size=args.batch_size)), data))
        # to make sure clients are in sync, sleep until next third minute
        sleep_until_next_3rd_minute()
    else:
        clear_cache()

    batch_count = 0

    with ExitStack() as stack:
        clients = initialize_clients(stack, args.url, args.model_name, args.limit)
        for i, wsi_path in enumerate(args.wsi_path):
            if args.numpy:
                iterator = iter(data)
            else:
                print(f"Processing {wsi_path}")
                hs_study = study(
                    wsi_path,
                    t=(args.tile_size, args.tile_size),
                    chunk=(
                        (args.chunk_size, args.chunk_size)
                        if args.chunk_size >= 0
                        else None
                    ),
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
                if args.inference_only:
                    data = list(map(lambda l: (np.copy(l[0].view()), l[1]), iterator))
                    iterator = iter(data)

            start_energy_usage_cpu, start_energy_usage_gpu = get_energy_reads(
                args.gpus, rapl_sockets
            )

            start_time = perf_counter()
            features, metadata, times, failures = track_method(
                inference_job,
                writer,
                live_tracking=args.live_tracking,
                path=wsi_path,
            )(iterator, clients=clients)
            end_time = perf_counter()
            end_energy_usage_cpu, end_energy_usage_gpu = get_energy_reads(
                args.gpus, rapl_sockets
            )
            elapsed_time = end_time - start_time
            write_tritonserver_metrics(args.metrics_endpoint, writer)
            assert len(failures) == 0, "should not be any failures"
            number_of_tiles = len(features)
            writer.add_scalar("number_of_tiles", number_of_tiles)
            number_of_batches = math.ceil(number_of_tiles / args.batch_size)
            writer.add_scalar("number_of_batches", number_of_batches)

            batch_count += plot_throughput(writer, times, batch_count)

            # Calculate throughput (tiles per second)
            throughput = number_of_tiles / elapsed_time
            total_throughput += number_of_tiles
            writer.add_scalar("throughput_tiles_per_second", throughput)
            print(f"Throughput: {throughput:.2f} tiles/second")

            # Calculate throughput (batches per second)
            throughput_batches = number_of_batches / elapsed_time
            writer.add_scalar("throughput_batches_per_second", throughput_batches)
            print(f"Throughput: {throughput_batches:.2f} batches/second")

            h, m, s = convert_seconds_to_hms(elapsed_time)
            # print(f"Number of images processed: {image_count}")
            print(f"Inference Time: {m} m, {s} s.")
            writer.add_scalar("inference_time_sec", h * 60 * 60 + m * 60 + s)
            total_time_used += elapsed_time

            write_energy_stats(
                start_energy_usage_cpu,
                end_energy_usage_cpu,
                start_energy_usage_gpu,
                end_energy_usage_gpu,
                elapsed_time,
                writer,
            )

    h, m, s = convert_seconds_to_hms(total_time_used)
    writer.add_scalar("inference_time_total_sec", h * 60 * 60 + m * 60 + s)

    total_tiles_s = total_throughput / total_time_used
    writer.add_scalar("throughput_total_tiles_per_second", total_tiles_s)
    print(f"Total throughput: {total_tiles_s} tiles/s")


if __name__ == "__main__":
    main()
    print("Done!")
