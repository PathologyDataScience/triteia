import pandas as pd
import argparse
import functools
import subprocess
import sys
import time

import tensorflow as tf
from large_image.cache_util import cachesClear

from simple_triton.config import *
from simple_triton.feature_extraction import study, inference
from simple_triton.model import TritonModel
from simple_triton.tile_iterators import TiffPrefetch
from simple_triton.utils import analyze


class Benchmark:
    """A class to benchmark inference server requests.

    The class takes as input args and performs histomic stream study, load model, histomics stream inference.
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict

    def create_hs_study(self, wsi_path, mask_path):
        """Create a histomic stream study.

        Parameters used in this cell are for reading
        from the whole-slide image (magnification, tile size, tile overlap, mask file).

        Args:
            wsi_path (string): path for .svs file
            mask_path (string): path for png file
        """
        # slide parameters
        tile = self.args_dict["tile"]
        # create a histomic-stream study from a wsi/mask pair
        self.hs_study = study(
            (wsi_path, mask_path),
            t=(tile, tile),
            chunk=(tile, tile),
            objective=20.0,
        )

    def create_load_model(self):
        """The function feature_extractor can be used to create feature extraction
        models in the model repository. Note - this cell will take time as the model
        is downloaded, saved, and loaded into triton.

        Parameters in this stage include the inference server (address), the model (model name,
        maximum batch size).

        """
        # slide parameters
        url = self.args_dict["url"]  # url for grpc access to triton server
        model_name = self.args_dict["model_name"]
        maxBatchSize = self.args_dict["maxbatchsize"]  # set max batch size
        count = self.args_dict["gpu_instance_count"]  # set gpu count
        kind = self.args_dict["kind"]  # set gpu kind
        gpus = self.args_dict["gpu_num"]  # set number of gpus
        model = TritonModel(model_name, url)
        model.load({"maxBatchSize": maxBatchSize})
        assert model.is_loaded()

        #  Add an instance group defining the model hardware resources and instances.
        if kind == "gpu":
            start, end, intval = 0, gpus, 1
            gpus = list(range(start, end, intval))
            instances = InstanceGroup(count=count, kind="gpu", gpus=gpus)
        else:
            instances = None

        optimization = TensorflowOptimization(
            amp=TensorflowMixedPrecision() if self.args_dict["use_amp"] else None
        )
        config = TensorflowConfig(
            name=args_dict["model_name"],
            max_batch_size=maxBatchSize,
            instance_group=instances,
            response_cache=False,
            optimization=optimization,
        )
        # load tensorflow model with larger batch size
        model.load(config=config.json())
        assert model.is_loaded()

    def inference_measure_throughput(self):
        """
        Run the inference and measure throughput on the Triton Inference Server.

        Parameters here include the number of tiles per batch, the number of workers,
        and the maximum number of pending inferences per worker.

        Returns:
            Inference time
        """

        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        # inference parameters
        batch = self.args_dict["batch"]
        model_name = self.args_dict["model_name"]
        limit = self.args_dict[
            "limit"
        ]  # limit on number of pending requests per worker
        workers = self.args_dict["workers"]  # total number of Submitter workers
        throughput = 0
        elapsed_time = 0
        num = 1
        throughput_results = []
        elapsed_time_results = []
        self.create_hs_study(
            self.args_dict["wsi_path"][0],
            self.args_dict["mask_path"][0],
        )
        dtype = np.uint8
        iterator = TiffPrefetch(
            self.hs_study, dtype=dtype, batch=batch, workers=workers
        )
        # warm up Model
        print("Warmup Model")
        (
            self.features,
            self.tile_info,
            self.times,
            self.failed,
        ) = inference(
            iterator,
            model_name,
            url=args_dict["url"],
            limit=limit,
        )
        # inference for number of iterations
        print("Total iterations:", self.args_dict["iterations"])
        for i in range(self.args_dict["iterations"]):
            print("Iteration:", num)
            self.cache_clear()
            self.gpu_mem_clear()
            print(
                "wsi_path: ",
                self.args_dict["wsi_path"][0] + "  mask_path: ",
                self.args_dict["mask_path"][0],
            )
            self.create_hs_study(
                self.args_dict["wsi_path"][0],
                self.args_dict["mask_path"][0],
            )
            iterator = TiffPrefetch(
                self.hs_study, dtype=dtype, batch=batch, workers=workers
            )
            # start timer
            start = time.time()
            (
                self.features,
                self.tile_info,
                self.times,
                self.failed,
            ) = inference(
                iterator,
                model_name,
                url=args_dict["url"],
                limit=limit,
            )
            elapsed_time_single = time.time() - start
            # throughput and elapsed time for single inference
            throughput_single = self.tile_info["version"].size / elapsed_time_single
            # Append to a list
            throughput_results.append(throughput_single)
            elapsed_time_results.append(elapsed_time_single)
            # Add throughput and elapsed time for multiple iterations
            throughput = throughput + throughput_single
            elapsed_time = elapsed_time + elapsed_time_single
            print("throughput single inference: ", throughput_single)
            print("Elapsed Time(sec): ", elapsed_time)
            print("\n")
            num += 1
        # Calculate average
        self.args_dict["throughput"] = throughput / (self.args_dict["iterations"])
        self.args_dict["elapsed_time"] = elapsed_time / (self.args_dict["iterations"])
        self.args_dict["throughput_all"] = throughput_results
        analyze(self.times)

    def client_nogpu(self):
        """Run client with no GPUs"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        assert len(tf.config.list_physical_devices("GPU")) == 0

    def cache_clear(self):
        """clearn the cache before running inference"""
        cachesClear()

        @functools.lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

        def gfg():
            fib.cache_clear()

        fib(30)
        # Before Clearing
        print(fib.cache_info())
        gfg()
        # After Clearing
        print(fib.cache_info())

    def gpu_mem_clear(self):
        """clearn the memory before running inference"""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def install():
    """Install dependencies for running benchmarking interface tool"""
    # install large_image with tile sources as prereq, check feature_extraction.ipynb in examples directory.
    # install simple_triton
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"../simple_triton"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ray"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])


def check_readiness(args_dict):
    """check readiness of models"""
    model = TritonModel(args_dict["model_name"], args_dict["url"])
    assert model.is_loaded()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=False,
        default="convnextsmall",
        help="Set model name, usage: --model-name=ConvNeXtXLarge or convnextsmall",
    )  # For testing, it will be removed
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Set inference batch size usage: --batch=64, default: 64",
    )  # For testing, it will be removed
    parser.add_argument(
        "--maxbatchsize",
        type=int,
        default=64,
        help="Set max batch size, usage: --maxbatchsize 64, default: 64",
    )
    parser.add_argument("--models-path", default="/tf/notebooks/models", required=False)
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use auto matic mixed precision, usage: --use-amp, default: False",
    )  # automatically creates a default value of False.
    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Use tensorRT, usage: --use-trt, default: False",
    )  # automatically creates a default value of False.
    parser.add_argument(
        "--precision",
        choices=["FP32", "FP16"],
        default="FP32",
        required=False,
        help="Choose between Precision FP16 or FP32 , default: FP32",
    )
    parser.add_argument(
        "--kind",
        choices=["gpu", "cpu"],
        default="gpu",
        required=False,
        help="choice between gpu or cpu, usage: --kind=gpu  default: gpu",
    )
    parser.add_argument(
        "--gpu-instance-count",
        type=int,
        default=1,
        required=False,
        help="number of instances for a gpu (default: 1), usage: --gpu-count=1",
    )
    parser.add_argument(
        "--gpu-num", default=8, type=int, help="Number of GPUs to use, default: 1"
    )
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--magnification", type=int, default=20)
    parser.add_argument("--tile", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=224)
    parser.add_argument("--mask-threshold", default=0.5)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("-v", "--verbose", default=True)
    parser.add_argument("-o", "--output", default="benchmark_out")
    # pass whole slide image
    parser.add_argument(
        "-w",
        "--wsi_path",
        nargs="+",
        help='<Required> Set flag, usage: python benchmark_interface.py -f "TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.svs","TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F_2.svs"',
        default=[
            "/tf/notebooks/TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.svs"
        ],
    )
    # pass binary mask image
    parser.add_argument(
        "-m",
        "--mask_path",
        nargs="+",
        help='<Required> Set flag, usage: python benchmark_interface.py -m "TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.mask.png","TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F_2.mask.png"',
        default=[
            "/tf/notebooks/TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.mask.png"
        ],
    )
    parser.add_argument(
        "--check-readiness", action="store_true"
    )  # check readiness of models
    parser.add_argument(
        "-i",
        "--iterations",
        default=1,
        type=int,
        help="Number of iterations of inference to check variation",
    )
    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    print(args_dict)
    print(f"Have", args_dict["wsi_path"])
    print(f"Have", args_dict["mask_path"])

    # show readiness if check is true
    if args_dict["check_readiness"] == True:
        check_readiness(args_dict)
        exit()  # exit after showing readiness
    # Add keras to model name
    keras_name = ".tensorflow"
    if args_dict["model_name"] == "ConvNeXtXLarge":
        args_dict["model_name"] = args_dict["model_name"] + keras_name  # set model_name
    if args_dict["model_name"] == "convnextsmall":
        args_dict["model_name"] = args_dict["model_name"] + keras_name  # set model_name
    benchmark = Benchmark(args_dict)
    benchmark.cache_clear()
    benchmark.client_nogpu()
    benchmark.gpu_mem_clear()
    benchmark.create_load_model()
    benchmark.inference_measure_throughput()
    # display elapsed time
    print(
        f"Throughput (tiles/sec):",
        args_dict["throughput"],
        "elapsed_time(sec):",
        args_dict["elapsed_time"],
    )

    # Create a directory if it doesn't exist for path in args_dict["output"]
    if not os.path.exists(args_dict["output"]):
        os.makedirs(args_dict["output"])

    # Create a dictionary with the information
    data = {
        "model_name": [args_dict["model_name"]],
        "maxbatchsize": [args_dict["maxbatchsize"]],
        "gpu_num": [args_dict["gpu_num"]],
        "gpu_instance_count": [args_dict["gpu_instance_count"]],
        "use_amp": [args_dict["use_amp"]],
        "use_trt": [args_dict["use_trt"]],
        "precision": [args_dict["precision"]],
        "workers": [args_dict["workers"]],
        "limit": [args_dict["limit"]],
        "iterations": [args_dict["iterations"]],
        "throughput(tiles/sec)": [args_dict["throughput"]],
        "elapsed_time(sec)": [args_dict["elapsed_time"]],
    }

    df = pd.DataFrame(data)
    out_csv = os.path.join(args_dict["output"], "benchmark_out.csv")
    df.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)

    throughput_data = {
        "throughput_all": args_dict["throughput_all"]
    }
    df_throughput = pd.DataFrame(throughput_data)
    # limit to two decimals
    df_throughput = df_throughput.round(2)
    throughput_fileoutput = os.path.join(args_dict["output"], "benchmark_throughput_out.csv")
    df_throughput.to_csv(throughput_fileoutput, mode='a', header=not os.path.exists(throughput_fileoutput), index=False)




