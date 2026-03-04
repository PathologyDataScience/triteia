import argparse
import functools
import gc
import os
import sys
from pathlib import Path

import pynvml


def convert_seconds_to_hms(seconds: float) -> tuple[int, int, int]:
    """Given time in seconds, return three integers representing hours,
    minutes, and seconds.

    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return hours, minutes, secs


def mock_metadata(batch_size):
    metadata = [
        {
            "version": b"version-1",
            "tile_height": 224,
            "tile_width": 224,
            "overlap_height": 0,
            "overlap_width": 0,
            "filename": b"/data/5/TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs",
            "slide_name": b"TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs",
            "slide_group": b"TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs",
            "chunk_width": 896,
            "chunk_height": 896,
            "target_magnification": 20.0,
            "scan_magnification": 40.0,
            "read_magnification": 40.0,
            "returned_magnification": 20.0,
            "level": 8,
            "slide_width": 54717,
            "slide_height": 45252,
            "slide_height_tiles": 202,
            "slide_width_tiles": 244,
            "chunk_top": 0,
            "chunk_left": 0,
            "chunk_bottom": 896,
            "chunk_right": 896,
            "tile_top": 0,
            "tile_left": 0,
        }
        for n in range(batch_size)
    ]
    return metadata


def clear_cache():
    # clear_cahce_shell_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'paper', 'clear_cache.sh')
    # os.chmod(clear_cahce_shell_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    # result = subprocess.run([f'{os.path.dirname(os.path.realpath(__file__))}/paper/clear_cache.sh'])
    # if result.returncode != 0:
    #     raise Exception(f"Failed to clear disk cache: {result.returncode}")
    # else:
    #     print(f"Disk cache cleared")

    gc.collect()

    # All objects collected that have LRU cache
    objects = [
        i for i in gc.get_objects() if isinstance(i, functools._lru_cache_wrapper)
    ]

    # All objects cleared that have LRU cache
    for object in objects:
        object.cache_clear()

    # Also use cachesClear utility to clear the cache that is managed by large_image special
    # cache wrapper
    from large_image.cache_util import cachesClear

    cachesClear()

    # Collect garbage for clearing all LRU cache as well
    gc.collect()

    print(f"LRU cache cleared")


def write_energy_stats(
    start_cpu_joules,
    end_cpu_joules,
    start_gpu_joules,
    end_gpu_joules,
    elapsed_time,
    writer,
):
    watts_concat = []
    joules_concat = []
    for i, (s, e) in enumerate(zip(start_gpu_joules, end_gpu_joules)):
        joules = e - s
        joules_concat.append(joules)
        watts = int(joules / elapsed_time)  # don't care about roundoff
        watts_concat.append(watts)
        writer.add_scalar(f"energy_usage_gpu_joule_{i}", joules)
        writer.add_scalar(f"energy_usage_gpu_watt_{i}", watts)

    writer.add_scalar(f"energy_usage_gpu_joule_total", sum(joules_concat))
    writer.add_scalar(f"energy_usage_gpu_watt_total", sum(watts_concat))

    watts_concat = []
    joules_concat = []
    for i, (s, e) in enumerate(zip(start_cpu_joules, end_cpu_joules)):
        joules = e - s
        joules_concat.append(joules)
        watts = int(joules / elapsed_time)
        watts_concat.append(watts)
        writer.add_scalar(
            f"energy_usage_cpu_joule_{i}",
            joules,
        )
        writer.add_scalar("energy_usage_cpu_watt", watts)

    writer.add_scalar(f"energy_usage_cpu_joule_total", sum(joules_concat))
    writer.add_scalar(f"energy_usage_cpu_watt_total", sum(watts_concat))


def get_energy_reads(gpus, rapl_sockets):
    gpu_joules = []
    for gpu in gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        joules = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 10e2)
        gpu_joules.append(joules)
    cpu_joules = []
    for rs in rapl_sockets:
        joules = int(read_power_uj(rs) / 10e5)
        cpu_joules.append(joules)

    return cpu_joules, gpu_joules


def read_power_uj(virt_file):
    try:
        with Path(virt_file).open("r") as f:
            return int(f.read().strip())
    except PermissionError as pe:
        print(f"Permission error reading {virt_file}: {pe}", file=sys.stderr)
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference. Not all parameters may be used in this script, see relevant code."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--huggingface-model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        type=str,
        required=False,
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--mag",
        type=int,
        default=20,
        help="Magnification level (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing (default: 128).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help="Max batch size for triton (default: 256).",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=16,
        help="Number of batches to prefetch (default: 16).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Total number of worker processes (default: 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
        help="The maximum number of allowable pending inferences. Default value 16. Recommened is 2xnum_gpu",
    )
    parser.add_argument(
        "--instance-group",
        type=int,
        default=1,
        help="Number of instance groups (default: 1).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Path to the tritonserver URL (default: localhost:8001).",
    )
    parser.add_argument(
        "--wsi-path",
        type=str,
        default="test_data/wsi/TCGA-AN-A0G0-01Z-00-DX1.svs",
        help="Path to the WSI file (default: test_data/wsi/TCGA-AN-A0G0-01Z-00-DX1.svs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./out",
        help="Path to the output directory (default: ./out).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Size of tiles in pixels (default: 224).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=896,
        help="Chunk size, calculated as tile_size * 2 (default: 896).",
    )
    parser.add_argument(
        "--icc",
        action="store_true",
        default=True,
        help="Apply ICC color correction (default: True).",
    )
    parser.add_argument(
        "--no-icc",
        dest="icc",
        action="store_false",
        help="Disable ICC color correction.",
    )

    parser.add_argument(
        "--manual-preload",
        dest="preload",
        action="store_false",
        help="Whether to expect the model to be preloaded or not (see script)",
    )

    parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Use numpy to generate random data instead of WSI data",
    )

    parser.add_argument(
        "--nchw",
        action="store_true",
        default=False,
        help="Emit NCHW tile batches from iterator instead of NHWC (default: False).",
    )

    parser.add_argument(
        "--inference-only",
        action="store_true",
        default=False,
        help="Whether or not to pre-load all tiles into memory before inference (default: False).",
    )

    parser.add_argument(
        "--no-live-tracking",
        action="store_false",
        dest="live_tracking",
        default=True,
        help="Whether to track live progress of inference.",
    )
    parser.add_argument(
        "--tensorboard-name",
        required=False,
        type=str,
        help="Name of the run, used for tracking stats. e.g. 'testing_no_cache', etc, or leave blank",
    )
    parser.add_argument(
        "--metrics-endpoint",
        required=False,
        type=str,
        default="http://localhost:8002/metrics",
        help="Tritonserver metrics endpoint. Used to scrape additional metrics for tensorboard",
    )
    # New argument: GPUs to track power consumption of
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=[],
        help="gpus to track power consumption of. If empty, no power consumption will be tracked.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.wsi_path):
        raise FileNotFoundError(f"WSI file {args.wsi_path} not found.")

    print(f"Using mag: {args.mag}")
    print(f"Using batch size: {args.batch_size}")

    if os.path.isfile(args.wsi_path):
        args.wsi_path = [args.wsi_path]
    else:
        args.wsi_path = sorted(
            list(
                map(
                    str,
                    list(Path(args.wsi_path).glob("*.svs"))
                    + list(Path(args.wsi_path).glob("*.tiff"))
                    + list(Path(args.wsi_path).glob("*.ndpi")),
                )
            )
        )
    if not args.wsi_path:
        raise FileNotFoundError(f"No WSI files found in {args.wsi_path}")

    if not os.path.exists(args.output_path):
        raise FileNotFoundError(f"Did not find output directory {args.output_path}")

    return args
