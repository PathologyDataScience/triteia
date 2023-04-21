import argparse
import tritonclient.grpc as grpcclient
from mil.io.utils import study
from google.protobuf.json_format import MessageToDict
from simple_triton.feature_extraction import feature_extractor
from simple_triton.model import model_config
import tritonclient.grpc as grpcclient
from simple_triton.feature_extraction import histomics_stream_inference
from simple_triton.submitter import analyze
from simple_triton.config import ConfigBuilder
import time
import subprocess
import sys
import json


class Benchmark:
    """A class to benchmark inference server requests.

    The class takes as input args performs histomic stream study, load model
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict

    def create_hs_study(self):
        """Create a histomic stream study.

        Parameters used in this cell are for reading
        from the whole-slide image (magnification, tile size, tile overlap, mask file).

        Args:
        args_dict (dict): The inputs to the model from argparse.
        tile (int): tile default value is 224.
        wsi_path (string): path for .svs file
        mask_path (string): path for png file
        """
        # slide parameters
        tile = self.args_dict["tile"]
        wsi_path = self.args_dict["wsi_path"]
        mask_path = self.args_dict["mask_path"]

        # create a histomic-stream study from a wsi/mask pair
        self.hs_study = study(
            (wsi_path, mask_path),
            t=(tile, tile),
            chunk=(tile, tile),
            target=20,
            source="exact",
        )

    def create_load_model(self):
        """The function feature_extractor can be used to create feature extraction
        models in the model repository. Note - this cell will take time as the model
        is downloaded, saved, and loaded into triton.

        Parameters in this stage include the inference server (address), the model (model name,
        maximum batch size).

        Args:
        client (tritonclient.grpc.InferenceServerClient):
        args_dict (dict): The inputs to the model from argparse.
        maxBatchSize (int): max batch size to for config
        """
        # slide paramters
        url = self.args_dict["url"]  # url for grpc access to triton server
        model_name = self.args_dict["model_name"]
        maxBatchSize = self.args_dict["maxbatchsize"]  # set max batch size
        verbose = self.args_dict["verbose"]  # set verbose
        count = self.args_dict["gpu_count"]  # set gpu count
        kind = self.args_dict["kind"]  # set gpu kind
        gpus = self.args_dict["gpu_num"]  # set number of gpus

        config = {"maxBatchSize": maxBatchSize}
        config = {"name": model_name}

        # create triton client
        client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

        config_builder = ConfigBuilder(
            config=config, client=client, model_name=model_name
        )

        # Add/remove an automatic mixed-precision accelerator to the config.
        if self.args_dict["use_amp"] == True:
            print("amp = True")
            config_builder.add_mixed_precision()
        elif self.args_dict["use_amp"] == False:
            config_builder.remove_instance_groups()
        # Add an TensorRT accelerator to the config.
        if self.args_dict["use_trt"] == True:
            config_builder.add_trt()
        elif self.args_dict["use_trt"] == False:
            config_builder.remove_trt

        #  Add an instance group defining the model hardware resources and instances.
        if kind == "gpu":
            start, end, intval = 0, gpus, 1
            gpus = list(range(start, end, intval))
            config_builder.add_instance_group(count, kind, gpus)
        if kind == "cpu":
            config_builder.add_instance_group(kind=kind)

        # load tensorflow model with larger batch size
        config_builder.max_batch_size(maxBatchSize)

        print(config)

        client.load_model(model_name, config=json.dumps(config))

        # check readiness
        client.get_model_repository_index()

        # deleting the client in main prevents conflicts with child process clients
        del client

    def inference_measure_throughput(self):
        """
        Run the inference and measure throughput on the Triton Inference Server.

        Parameters here include the number of tiles per batch, the number of workers,
        and the maximum number of pending inferences per worker.

        Args:
            client (tritonclient.grpc.InferenceServerClient):
            args_dict (dict): The inputs to the model from argparse.

        Returns:
            Inference time
        """
        # inference parameters
        batch = self.args_dict["batch"]
        model_name = self.args_dict["model_name"]
        limit = self.args_dict[
            "limit"
        ]  # limit on number of pending requests per worker
        workers = self.args_dict["workers"]  # total number of Submitter workers

        # start timer
        start = time.time()

        # inference
        self.features, self.tile_info, self.times = histomics_stream_inference(
            self.hs_study,
            model_name,
            args_dict["url"],
            batch=batch,
            workers=workers,
            limit=limit,
        )

        elapsed_time = {time.time() - start}
        analyze(self.times)
        return elapsed_time

    def client_noGPU(self):
        """Run client with no GPUs

        If running Triton and the client on the same machine, we want to stop the client tensorflow
        from consuming GPU resources. By default, TensorFlow maps nearly all available GPU memory.
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import tensorflow as tf

        assert len(tf.config.list_physical_devices("GPU")) == 0

    def cache_clear(self):
        import functools

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
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def install():
    # install large_image with tile sources as prereq, check feature_extraction.ipynb in examples directory.
    # install simple_triton
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"../simple_triton"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ray"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
    # install mil
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"../../mil"])


def check_readiness():
    """check readiness of models"""
    # create triton client
    url = "localhost:8001"  # url for grpc access to tirton server
    client = grpcclient.InferenceServerClient(url=url, verbose=True)
    # check readiness
    client.get_model_repository_index()
    del client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", required=False, default="ConvNeXtXLarge", help="Set model name, usage: --model-name=ConvNeXtXLarge, default: ConvNeXtXLarge"
    )  # For testing, it will be removed
    parser.add_argument(
        "--batch", type=int, default=64, help="Set inference batch size usage: --batch=64, default: 64"
    )  # For testing, it will be removed
    parser.add_argument("--maxbatchsize", type=int, default=256, help="Set max batch size, usage: --maxbatchsize=256, default: 256")
    parser.add_argument("--models-path", default="/tf/notebooks/models", required=False)
    parser.add_argument(
        "--use-amp", action="store_true", help="Use auto matic mixed precision, usage: --use-amp, default: False"
    )  # automatically creates a default value of False.
    parser.add_argument(
        "--use-trt", action="store_true", help="Use tensorRT, usage: --use-trt, default: False"
    )  # automatically creates a default value of False.
    parser.add_argument(
        "--precision", choices=["FP32", "FP16"], default="FP16", required=False, help="choose between Precision FP16 or FP32 , default: FP16"
    )
    parser.add_argument(
        "--kind",
        choices=["gpu", "cpu"],
        default="gpu",
        required=False,
        help="choice between gpu or cpu, usage: --kind=gpu  default: gpu"
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        required=False,
        help="number of instances for a gpu (default: 1), usage: --gpu-count=1",
    )
    parser.add_argument(
        "--gpu-num", default=1, type=int, help="Number of GPUs to use, default: 1"
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
    parser.add_argument(
        "--check-readiness", action="store_true"
    )  # check readiness of models
    args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    print(args_dict)
    if args_dict["check_readiness"] == True:
        check_readiness()
        exit()  # exit after showing readiness
        # Add keras to model name
    keras_name = ".tensorflow"
    if args_dict["model_name"] == "ConvNeXtXLarge":
        args_dict["model_name"] = args_dict["model_name"] + keras_name  # set model_name
    args_dict[
        "wsi_path"
    ] = "/tf/notebooks/TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.svs"
    args_dict[
        "mask_path"
    ] = "/tf/notebooks/TCGA-AN-A0G0-01Z-00-DX1.BE0BB5DF-DEDA-48D8-B5D8-2735C767F28F.mask.png"

    # Install dependencies
    # install() # uncomment to install dependencies
    benchmark = Benchmark(args_dict)
    benchmark.cache_clear()
    benchmark.client_noGPU()
    benchmark.gpu_mem_clear()
    benchmark.create_hs_study()
    benchmark.create_load_model()
    elapsed_time = benchmark.inference_measure_throughput()

    # display elapsed time
    print(f"Total elapsed time:", elapsed_time)
