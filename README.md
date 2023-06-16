# simple-triton

A simple Python client for efficient inference with the NVIDIA Triton inference server.

See the [user guide](#user-guide) to read about concepts and to get started with examples. Details on testing and implementation are located in the [developer guide](#developer-guide).

## Supported Triton version
simple-triton is tested with [Triton version 23.04](https://github.com/triton-inference-server/server/releases/tag/v2.33.0).

# User guide <a name="user-guide"></a>

## Contents

- [Quick start](#quick-start)
    - [Example](#example)
    - [Running the Triton container](#container)
- [Triton concepts](#concepts)
    - [Model control](#control)
    - [Model configuration](#config)
- [Developer guide](#developer-guide)
    - [Testing](#testing)
    - [Benchmarking](#benchmarking)

## Quick start <a name="quick-start"></a>

simple-triton is pip installable. Use of the whole slide image reader requires installation of `histomcs_stream` and `large_image` with tiff or openslide tile sources 
```
sudo apt update
sudo apt install -y python3-openslide openslide-tools
pip install histomics_stream 'large_image[tiff]' \
  scikit_image --find-links https://girder.github.io/large_image_wheels
```

### Example <a name="example"></a>

The notebook `examples\feature_extraction.ipynb` demonstrates whole-slide image feature extraction. This example requires installation of the `mil` library and a running Triton container on the client machine.

### Running the Triton container <a name="container"></a>

The simplest way to deploy Triton is to run a Triton Docker container from the Nvidia GPU Cloud (NGC). See the NVIDIA Triton [quick start guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html) for more information on running and verifying the container, including specifying a model repository.

Two Triton server container runtime options are important for use with simple-triton:
1. `--model-control-mode=explicit` is required to be able to load and modify models at runtime
2. `--strict-model-config=false` allows Triton to auto-fill many options for model configuration

for example,

```
docker run --gpus=8 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 -v host_model_repository:/models nvcr.io/nvidia/tritonserver:22.05-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false --strict-model-config=false
```

To use shared memory for client/server communication, both the client and server containers must be run with `--ipc=host`. Additionally, the client container should be run with the `--shm-size` option to request an expansion of the default 64MB shared memory. Running the client container with `--network=host` is the easiest configuration to allow the client and server to communicate over the host network.

## Triton concepts <a name="concepts"></a>
simple-triton is a Python client that simplifies the loading and configuration of models on the NVIDIA Triton Inference Server, as well as inference requests. Communication between client and server uses the Remote Procedure Call (gRPC) protocol. If the client and server share memory then shared memory can further accelerate communication. For inference tasks that are preprocessing intensive or I/O bound, simple-triton can be used with a multiprocessing data loader to shard loading, preprocessing, and inference requests over multiple processes.

The motivation for this project was to accelerate inference-intensive processes like feature extraction from whole-slide images. Machine-learning frameworks like TensorFlow or PyTorch that are primarily intended for model training are not optimal for large inference tasks. Triton provides several advantages over these frameworks including better utilization of hardware. Triton also decouples data loading and preprocessing of inference which improves flexibility in implementing these steps.

Some key performance optimizations of Triton that available through simple-triton:
1. Model concurrency - a single GPU can host multiple instances of a model using CUDA streams, allowing overlap of host/device communication and device computation.
2. Automatic mixed precision - simple-triton can enable mixed precision for TensorFlow models, improving throughput and GPU memory consumption.
3. TensorRT - take advantage of quantization, layer and tensor fusion, and kernel tuning for select model types.
4. Shared memory communication - send data to and receive data from Triton using shared memory when the client and server are on the same machine.

## Model control <a name="control"></a>
The `TritonModel` class can be used to load/unload models, to retrieve model configurations or metadata, or to check model if a mode is idle or loaded. A model is defined by a model name and server url

```python
from simple_triton.model import TritonModel
model = TritonModel("EfficientNetV2S.tensorflow", "localhost:8001")
```

When unloading a model, simple-triton will check that the model is idle

```python
# load model with auto-generated configuration
# block and timeout after 1 second
model.load(block=True, timeout=1.)
```

Model status can also be checked

```python
# check if model is loaded
assert model.is_loaded()

# check if model is idle
assert model.is_idle()
```

The current hosted model configuration can also be queried

```python
model.get_config()
```

## Model configuration <a name="config"></a>
The `ConfigBuilder` class is an interface to define hardware resources, optimizations, and model inputs/outputs for a model. This configuration can be used with the model control functions to alter the configuration of hosted models.

Each model should have a maximum batch size that defines the upper limit on the number of samples in a single request

```python
from simple_triton.config import ConfigBuilder
builder = ConfigBuilder("EfficientNetV2S.tensorflow")

# set max batch size to 64
builder.max_batch_size(64)
model.load(config=builder.config)

# for non-batching models, set batch size to zero
builder.max_batch_size(0)
nonbatching_model.load(config=builder.config)
```

Triton also has an inference results cache (should be disabled for benchmarking)

```python
# disabe inference result cache
builder.response_cache(False)
model.load(config=builder.config)
```

Automatic mixed precision or TensorRT optimization can also be enabled

```python
# enable automatic mixed precision
builder.add_mixed_precision()
model.load(config=builder.config)

# remove AMP and add TensorRT FP16
builder.remove_mixed_precision()
builder.add_trt("FP16")
model.load(config=builder.config)
```

Assignment of hardware resources is handled with *instance groups*. Each group defines CPU and GPU resources, and the number of concurrently hosted models on each resource. By default, Triton will create a single model instance on each GPU. 

```python
# add a second concurrent model instance on each GPU
builder.add_instance_group(count=2, kind="GPU")
model.load(config=builder.config)

# restrict to GPUs 0, 1, 2, 3
builder.add_instance_group(count=2, kind="GPU", gpus=[0, 1, 2, 3])
model.load(config=builder.config)

# add CPU instances to the GPU instances
builder.add_instance_group(count=2, kind="CPU")
model.load(config=builder.config)

# remove all instances groups and return to default 1 instance / gpu for all GPUs
builder.remove_instance_groups()
model.load(config=builder.config)
```

The dimensions of model inputs and outputs can also be altered using `ConfigBuilder` methods. This is helpful when dealing with models that have variable-sized inputs/outputs that can be misinterpreted by Triton during loading.

# Developer guide <a name="developer-guide"></a>

## Testing <a name="testing"></a>

Testing and code formatting is automated using tox and pytest and can be run using `python -m tox run`. Running this will evaluate the tests in the environments defined in `tox.ini` and will format the source using Black. Following testing, a coverage.html file will be located in .tox/coverage.

Testing requires running a Triton server on the local machine. Tests are run using a `EfficientNetV2S.tensorflow` model that can be downloaded using pooch:

```python
import pooch
pooch.retrieve(
    fname="EfficientNetV2S.tensorflow.zip",
    url="https://drive.google.com/uc?export=download&id=1Mmm2sRGzdzCEAODjABiiPIiBdg40EPwC&confirm=t&uuid=b11e409a-64b2-4146-b45d-4f229093cb5a&at=ANzk5s7UvBzB7zpqm7AvngovJwS8:1681783040828",
    known_hash="a6ed53d8343498b4ebfe7ff1a9ccbcabef23d6a164d2a521916774af49996f7e",
    path=host_model_repository
)
```
## Benchmarking <a name="benchmarking"></a>

We provide a benchmark command-line tool to help users identify the most performant parameters for their inference projects. An exhaustive list of evaluable parameters is provided below and includes batch size, precision accelerators, concurrent instances, and workers for data loading and preprocessing. This allows users to quickly tailor parameters to their specific model and dataset given available system resources and environment requirements. For inference jobs that may span days, a few hours of benchmarking can be a worthwhile investment.

In contrast to NVIDIA's [Triton Model Anlyzer](https://github.com/triton-inference-server/model_analyzer) which uses randomly generated arrays for benchmarking, our tool incorporates a whole-slide image dataloader to introduce real IO conditions for more realistic measurements. We also provide options for enabling accelerators like mixed precision or TensorRT conversion. 

With default parameters, the benchmark uses 32 workers to read and batch tiles from a whole-slide image using the [large_image reader](https://github.com/girder/large_image). Each worker maintains a queue of maximum 10 inference requests with 64 tiles per batch and 1 batch per inference request. To explore additional parameters users can provide additional command line arguments or override default values
 
```
python /tf/notebooks/simple_triton/benchmarking/benchmark_interface.py  --gpu-num $gpu_num  --use-trt --precision "FP16" --fileoutput $filename   --iterations 5  --maxbatchsize $maxbatchsize  --model-name "ConvNeXtXLarge"
```

### Output

A single benchmark print the results as dict to parse easily. Results as dict are also store the result in file format "benchmark_output.txt". Set of features stored along with throughput and time elaped are model_name, maxbatchsize, gpu_num, gpu_count, use_amp, use_trt, precision, workers, limit, throughput, elapsed_time. 
An example output for an experiment run shows,
```
model_name: convnextsmall.tensorflow,  Max Batch Size: 64, gpu-num: 8, instance group count: 1, amp: False, trt: False, precision: FP16, workers: 32, Limit: 10, throughput: 195.95287948015198, elapsed_time: 28.640860160191853 
```

Output  also shows detailed time taken in sec as median, min, max for values such as total, data loading, results return, in-process, completion,  retrieval and other factors. An example output show,

```
                             median    min    max
-------------------------  --------  -----  -----
total (sec)                    4.01   1.37   7.97
data loading (% total)        57.63  35.97  83.60
results return (% total)       0.14   0.05   0.62
in-process (% total)          42.24  16.05  63.92
completion (% in-process)     27.59   9.28  96.57
retrieval (% in-process)      70.85   0.36  89.94
other (% in-process)           1.68   0.72   7.13
```

### Scripting tool

Users can also run benchmarking sessions using calling the example benchmarking script with command-line arguments to run throughput multiple set of features and find the optimal results. A sample shell script in the example folder named "ConvNeXtXLarge_amp_Batch64_GPU8_iter5_BatchTest.sh", illustrates the use of the benchmarking tool to do a parameter sweep over the number of GPUs, and maximum batch size. Users can modofy the script to add/remove set of features and update their value. 
Each call to the benchmarking tool runs a warmup before making a series of measurements with the desired parameter settings. Caches are cleared between each measurement to ensure that measured throughput reflects real IO conditions, and that the inference results are not cached by Triton. Output of scripting tool is similar to actual output for eachb run. 


```
Explanation of each args is as follows,
<pre>
--model-name:       Set model name, usage: `--model-name ConvNeXtXLarge`
--fileoutput:       output file name with path for results
--batch:            Set inference batch size usage: `--batch 64`, default: 64
--maxbatchsize:     Set max batch size, usage: `--maxbatchsize 64`, default: 64
--use-amp:          Use auto matic mixed precision, usage: `--use-amp`, default: False
--use-trt:          Use tensorRT, usage: `--use-trt`, default: False
--precision:        Choose between Precision FP16 or FP32, usage: `--precision "FP16"`, default: FP16
--kind:             choice between gpu or cpu, usage: `--kind gpu`  default: gpu
--gpu-count:        number of instances for a gpu (default: 1), usage: `--gpu-count 1`
--gpu-num:          Number of GPUs to use, usage: `--gpu-num 2`, default: 1
--url:              url for connecting with Triton Inference Server, usage: `--url: "localhost:8001"`, default=localhost:8001
--magnification:    Set magnification size, usage: `--magnification 20`, type=int, default=20
--tile:             Set tile size, usage `--tile 224`, type=int, default=224
--limit:            In the consumer we limit the number of pending requests to avoid flooding the inference server. 
                    type=int, usage `--limit 10`, default=10
--workers:          worker maintains a max queue of inferences. Worker return result via multiprocessing.queue
                    type=int, usage `--workers 32`, default=32
--iterations:       Number of iterations to perform inference on a single file
--wsi-path          file name and path for WSI image
--mask-path         File name and path of binary mask image
--check-readines:   check readiness of models. usage: `--check-readines`, default: false
</pre>
