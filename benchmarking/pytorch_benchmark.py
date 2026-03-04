"""
This code will do inference on a given folder or WSI file.
The code is written to be similar to how inference would be done by the triton client.

The script will print some stats to console, otherwise, the main idea is that data is written to tensorboards
"""
import glob
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from sys import path
from time import perf_counter

import numpy as np
import pynvml
import timm
import torch
from tensorboardX import GlobalSummaryWriter
from timm.data import create_transform, resolve_data_config
from transformers import (
    AutoImageProcessor,
    ResNetModel,
)

path.append(os.path.join(os.path.dirname(__file__), "../simple_triton"))
from simple_triton.feature_extraction import study
from simple_triton.tile_iterators import TiffPrefetch
from util import (
    clear_cache,
    convert_seconds_to_hms,
    parse_args,
    get_energy_reads,
    write_energy_stats,
)

from simple_triton.utils import init_tb_writer, track_method


def normalize_image(in_0, device, mean, std):
    in_t = torch.from_numpy(in_0).to(device, dtype=torch.float16)
    in_t = (in_t / 255.0).permute(0, 3, 1, 2)
    return (in_t - mean) / std


def work_normalized(args):
    (model, processor, dtype, device, mean, std), batch = args

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.float16),
    ):
        start_inference = perf_counter()
        in_t = processor(batch[0].view(), device, mean, std)
        features = model(in_t).cpu().numpy()
        end_inference = perf_counter()

    latency_ms = (end_inference - start_inference) * 1000
    return features.shape[0], latency_ms, features, batch[1]


def work(args):
    (model, processor, dtype, device, _, __), batch = args
    with torch.inference_mode():
        start_inference = perf_counter()
        inputs = processor(batch[0].view(), device=device, return_tensors="pt")
        output = model(**inputs).pooler_output.flatten(1, -1).cpu().numpy()
        end_inference = perf_counter()

        # Calculate latency in milliseconds
        latency_ms = (end_inference - start_inference) * 1000
    return output.shape[0], latency_ms, output, batch[1]


def warmup_models(worker_params, tile_size, model_name):
    n = len(worker_params)

    test_data = np.random.randn(10, 64, tile_size, tile_size, 3).astype(np.uint8)
    metadata = [{"key": "value", "tile_left": 0}] * test_data.shape[2]
    data = list(map(lambda l: (l, metadata), test_data))
    test_iterator = iter(data)

    def task_iter():
        for _n, item in enumerate(test_iterator):
            yield worker_params[_n % n], item

    with ThreadPoolExecutor(n) as executor:
        results = []
        if model_name.upper() == "RESNET50":
            futs = [executor.submit(work, arg) for arg in task_iter()]
        elif model_name.upper() == "UNI" or model_name.upper() == "GIGAPATH":
            futs = [executor.submit(work_normalized, arg) for arg in task_iter()]

        total_tiles = 0
        latencies = []
        for f in as_completed(futs):
            num_tiles, latency, output, meta = f.result()
            total_tiles += num_tiles
            latencies.append(latency)
    return total_tiles, latencies


def main():
    args = parse_args()

    """
    How to implement a comparable multiprocessing framework to triton's inference?
    
    triton's workflow: multiple workers/threads fetch image data. An iterator spawns a thread for each returned batch of tiles which is queued for execution remotely
    
    pytorch's traditional workflow is different: each GPU has it's own thread where it does all the processing. So there is no shared iterator.
    Option A: to use pytorch's DDP: divide up the `hs_study`'s "read_kwargs" and initialize separate TiffPretech workers per thread
        Upside: This is the most conventional way to use Pytorch DDP.
                Code is probably easier to write and understand
        Downside: the workflow is different from triton. With multiple TiffPrefetch instances, it's hard to estimate IO gains.
                  also, it would be nice to avoid torchrun.
    """

    time_string = datetime.now().strftime("%y%m%d-%H%M%S")
    tb_name = (
        "pytorch_" + time_string if not args.tensorboard_name else args.tensorboard_name
    )
    writer: GlobalSummaryWriter = init_tb_writer(
        args.output,
        tb_name,
        [args.wsi_path],
        {
            "icc": args.icc,
            "workers": args.workers,
            "batch": args.batch_size,
            "prefetch": args.prefetch,
            "chunk": args.chunk_size,
            "tile_size": args.tile_size,
            "magnification": args.mag,
            "instance-group": args.instance_group,
            "wsi-path": args.wsi_path,
            "inference_only": args.inference_only,
            "cuda_available": torch.cuda.is_available(),
        },
    )

    image_processor = None
    model = None

    rapl_sockets = glob.glob(
        "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:*/energy_uj"
    )

    if not torch.cuda.is_available():
        print("USING ONE CPU THREAD ONLY!!!")
        args.gpus = [0]
    else:
        pynvml.nvmlInit()

    worker_params = []

    if args.model_name.upper() == "RESNET50":
        for gpu in args.gpus:
            device = (
                torch.device(f"cuda:{gpu}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device).eval()
            processor = AutoImageProcessor.from_pretrained(
                "microsoft/resnet-50", use_fast=True
            )
            worker_params.append((model, processor, None, device, None, None))
    elif args.model_name.upper() == "GIGAPATH":
        # login(
        #     os.getenv("HF_TOKEN")
        # )  # User Access Token, found at https://huggingface.co/settings/tokens
        for gpu in args.gpus:
            device = (
                torch.device(f"cuda:{gpu}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model = (
                timm.create_model(
                    "hf-hub:prov-gigapath/prov-gigapath",
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=True,
                )
                .to(device)
                .eval()
            )
            mean = (
                torch.tensor(model.pretrained_cfg["mean"]).view(1, 3, 1, 1).to(device)
            )
            std = torch.tensor(model.pretrained_cfg["std"]).view(1, 3, 1, 1).to(device)
            image_processor = create_transform(
                **resolve_data_config(model.pretrained_cfg, model=model)
            )
            worker_params.append((model, normalize_image, None, device, mean, std))
    elif args.model_name.upper() == "UNI":
        # login(
        #     os.getenv("HF_TOKEN")
        # )  # User Access Token, found at https://huggingface.co/settings/tokens
        for gpu in args.gpus:
            device = (
                torch.device(f"cuda:{gpu}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model = (
                timm.create_model(
                    "hf-hub:MahmoodLab/uni",
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=True,
                )
                .to(device)
                .eval()
            )
            mean = (
                torch.tensor(model.pretrained_cfg["mean"]).view(1, 3, 1, 1).to(device)
            )
            std = torch.tensor(model.pretrained_cfg["std"]).view(1, 3, 1, 1).to(device)
            worker_params.append((model, normalize_image, None, device, mean, std))
    else:
        print(f"Model name '{args.model_name}' not recognized.")

    total_time_used = 0
    total_throughput = 0
    batch_count = 0
    tile_count = 0

    clear_cache()

    print("Warming up model")
    t, _ = warmup_models(worker_params, args.tile_size, args.model_name)
    print(f"warmed up with {t} tiles")

    total_tiles = 0
    for i, wsi_path in enumerate(args.wsi_path):
        hs_study = study(
            wsi_path,
            t=(args.tile_size, args.tile_size),
            chunk=(args.chunk_size, args.chunk_size),
            objective=args.mag,
        )

        # Use sharded iterator so each rank only sees its subset of tiles
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

        n = len(worker_params)

        def task_iter():
            for _n, item in enumerate(iterator):
                yield worker_params[_n % n], item

        def iterate_image():
            with ThreadPoolExecutor(n) as executor:
                results = []
                if args.model_name.upper() == "RESNET50":
                    futs = [executor.submit(work, arg) for arg in task_iter()]
                elif (
                    args.model_name.upper() == "UNI"
                    or args.model_name.upper() == "GIGAPATH"
                ):
                    futs = [
                        executor.submit(work_normalized, arg) for arg in task_iter()
                    ]

                tiles = 0
                batches = 0
                latencies = []
                outputs = []
                metadata = defaultdict(list)
                for f in as_completed(futs):
                    num_tiles, latency, output, meta = f.result()
                    batches += 1
                    tiles += num_tiles
                    latencies.append(latency)
                    outputs.append(output)
                    for k, v in metadata.items():
                        metadata[k].append(v)

            return tiles, latencies, batches, np.concatenate(outputs, axis=0), metadata

        start_energy_usage_cpu, start_energy_usage_gpu = get_energy_reads(
            args.gpus, rapl_sockets
        )

        start_time = perf_counter()
        tiles_iterated, latencies, batches_iterated, features, metadata = track_method(
            iterate_image, writer, live_tracking=args.live_tracking
        )()
        end_time = perf_counter()
        end_energy_usage_cpu, end_energy_usage_gpu = get_energy_reads(
            args.gpus, rapl_sockets
        )
        elapsed_time = end_time - start_time

        writer.add_scalar("number_of_tiles", tiles_iterated)
        writer.add_scalar("number_of_batches", batches_iterated)

        # Calculate throughput (tiles per second)
        throughput = tiles_iterated / elapsed_time
        writer.add_scalar("throughput_tiles_per_second", throughput)
        print(f"Throughput: {throughput:.2f} tiles/second")

        # Calculate throughput (batches per second)
        throughput_batches = batches_iterated / elapsed_time
        writer.add_scalar("throughput_batches_per_second", throughput_batches)
        print(f"Throughput: {throughput_batches:.2f} batches/second")

        for l in latencies:
            writer.add_scalar("latency_ms", l)
        total_tiles += tiles_iterated
        writer.add_scalar("latency_mean_ms", np.mean(latencies))

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
    print(f"Seconds used: {total_time_used}")
    writer.add_scalar("inference_time_total_sec", h * 60 * 60 + m * 60 + s)
    total_tiles_s = total_tiles / total_time_used
    writer.add_scalar("throughput_total_tiles_per_second", total_tiles_s)
    print(f"Total throughput: {total_tiles_s} tiles/s")


if __name__ == "__main__":
    main()
    print("Done!")
