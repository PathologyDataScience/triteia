# simple-triton

A Python client for NVIDIA Triton inference.

See the [user guide](#user-guide) to read about concepts and to get started with examples. Details on testing and implementation are located in the [developer guide](#developer-guide).

## Supported Triton version
simple-triton is tested with [Triton version 23.03](https://github.com/triton-inference-server/server/releases/tag/v2.32.0).

# User guide <a name="user-guide"></a>

## Contents

- [Quick start](#quick-start)
    - [Example](#example)
    - [Running the Triton container](#container)
- [Triton concepts](#concepts)
    - [Model control](#control)
    - [Model configuration](#config)
- [Command-line interfaces](#cli)
    - [Inference](#inference)
- [Developer guide](#developer-guide)
    - [Testing](#testing)
    - [Benchmarking](#benchmarking)

## Quick start <a name="quick-start"></a>

simple-triton requires `histomcs_stream` and `large_image` packages with the tiff reader
```
git clone https://github.com/PathologyDataScience/simple_triton.git
pip install ./simple_triton histomics_stream 'large_image[tiff]'
```

Or, you can try the docker image:
```bash
# optional: download test data
python download_test_data.py

# simple_triton_client will be the name of the docker image
docker build . -t simple_triton_client:latest
docker run --init --security-opt seccomp:unconfined --network=container:<name of tritonserver docker image> --shm-size=1g -v ${PWD}/test_data:/data:ro --rm --name tritonclient --gpus all -it simple_triton_client:latest
```
> **_NOTE:_**  `--init` ensures that the docker container has a "master process" to do clean multi-processing. `--network=` lets the docker image see ports from other containers, in this case the triton server. The default shared memory size is now 64MB, so `--shm-size=` is necessary if you are reading large WSIs. `--security-opt seccomp:unconfined` might only be necessary on bigger machines, but it gives your process access to [openblas](https://www.openblas.net/) threads. `--rm` removes the container on exit, beware.


### Example <a name="example"></a>

The notebook `examples\feature_extraction.ipynb` demonstrates whole-slide image feature extraction. This example requires installation of the `mil` library and a running Triton container on the client machine.

### Running the Triton container <a name="container"></a>

The simplest way to deploy Triton is to run a Triton Docker container from the Nvidia GPU Cloud (NGC). See the NVIDIA Triton [quick start guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html) for more information on running and verifying the container, including specifying a model repository.

Two Triton server container runtime options are important for use with simple-triton:
1. `--model-control-mode=explicit` is required to be able to load and modify models at runtime
2. `--strict-model-config=false` allows Triton to auto-fill many options for model configuration

for example,

```
docker run --gpus=8 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 --shm-size=1g --ulimit memlock=-1 --ipc=host -v host_model_repository:/models nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false --strict-model-config=false
```

where `host_model_repository` is the location of your model repository folder on the host machine.

> **Note:** The options `--ipc`, `--shm-size`, and `--ulimit memlock` are recommended when using shared memory for client/server communication. This allows Triton to access host system shared memory, increases the default 64MB shared memory limit, and prevents paging of RAM out to disk. If the client is run in a container then the `--ipc` and `--shm-size` options should be passed to the client container run command. Running the client container with `--network=host` is the easiest configuration to allow the client and server to communicate over the host network.

## Command-line interface <a name="cli"></a>
### Inference <a name="inference"></a>
A command-line interface is provided for inference with single or multiple slides and with control of tiling, masking, data loading, and serialization parameters. Models must be loaded prior to inference.

Perform inference with the EfficientNetV2S model on a single slide, outputing serialized embeddings to your home directory
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow
```

Optional parameters allow restricting inference to a tissue mask (`-m`)
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -m TCGA-AN-A0G0-01Z-00-DX1.mask.png
```

modification tile size (`-t`), add tile overlap (`-o`), and change magnification (`-M`)
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -t 256 -o 128 -M 10
```

adjustment of tile reading parameters including ICC correction (`-i`), read chunk size (`-c`), batch size (`-b`), prefetch (`-p`), and multiprocessing workers (`-w`).
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -i -c 8 -b 128 -p 2 -w 16
```

Provide image source (`-n`) and target (`-r`) parameters for Macenko color normalization
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -n ~/TCGA-AN-A0G0-01Z-00-DX1.stain.npy -r ~/standard_stain.npy
```

Change the address of the Triton inference server
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -a "foo.edu:8001"
```

Increase the precision of serialized features to float (default is half float)
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -f
```

For large jobs use a tab-delimited file containing input images and optionally their masks and normalization stain profiles
```console
$more ~/inputs.tsv
TCGA-AN-A0G0-01Z-00-DX1.svs    TCGA-AN-A0G0-01Z-00-DX1.mask.py    TCGA-AN-A0G0-01Z-00-DX1.stain.npy    
TCGA-AN-A0G0-01Z-00-DX2.svs    TCGA-AN-A0G0-01Z-00-DX2.mask.py    TCGA-AN-A0G0-01Z-00-DX2.stain.npy
TCGA-AN-A0G0-01Z-00-DX3.svs    TCGA-AN-A0G0-01Z-00-DX3.mask.py    TCGA-AN-A0G0-01Z-00-DX3.stain.npy
TCGA-AN-A0G0-01Z-00-DX4.svs    TCGA-AN-A0G0-01Z-00-DX4.mask.py    TCGA-AN-A0G0-01Z-00-DX4.stain.npy
$python feature_extraction.py ~/inputs.tsv ~/ EfficientNetV2S.tensorflow
```

Skip images where output already exists
```console
$python feature_extraction.py ~/inputs.tsv ~/ EfficientNetV2S.tensorflow -s
```

## Triton concepts <a name="concepts"></a>
This client simplifies the loading and configuration of models and submitting inference requests on the NVIDIA Triton Inference Server. Remote Procedure Call (gRPC) protocol is used for efficient communication between client and server. For inference tasks that are preprocessing intensive or I/O bound, simple-triton can be used with a multiprocessing data loader to accelerate loading and preprocessing tasks like the application of color correction or normalization.

The client is built specifically for inference-intensive tasks like embedding tiles from whole-slide images. Machine-learning frameworks like TensorFlow or PyTorch that are primarily intended for model training are often suboptimal for inference. Triton provides several advantages including better utilization of hardware and better flexibility in data loading and preprocessing.

Benefits of Triton include:
1. Model concurrency - a single GPU can host multiple instances of a model using CUDA streams, improving the overlap of host/device communication and device computation.
2. Automatic mixed precision - simple-triton can enable mixed precision for TensorFlow and ONNX models, improving throughput and GPU memory consumption.
3. TensorRT - take advantage of quantization, layer and tensor fusion, and kernel tuning for TensorFlow and ONNX models.
4. Shared memory communication - send data to and receive data from Triton using shared memory when loading data directly on the server.

## Model control <a name="control"></a>
The `TritonModel` class can be used to load/unload models, to retrieve model configurations or metadata, or to check model if a model is idle or loaded. A model is defined by a model name and server url

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

# Inference <a name="inference"></a>

Inference is performed using `inference.inference`. This function consumes data from a simple iterator that emits data/metadata pairs. For single input models, data is provided as a numpy array. For multi-input models, data is provided as a dict of key value pairs linking numpy arrays to model input names (visible from `Model.get_config()`).

`inference` can apply preprocessing functions to data after loading and prior to inference by passing a callable to the `pre` argument. 

> **Note:** For multi-input models all inputs are required to uniform batch dimensions. Duplicate singleton values where necessary to satisfy this requirement.

# Developer guide <a name="developer-guide"></a>

## Testing <a name="testing"></a>

Testing and code formatting is automated using tox and pytest and can be run using `python -m tox run`. Running this will evaluate the tests in the environments defined in `tox.ini` and will format the source using Black. Following testing, a coverage.html file will be located in .tox/coverage.

Testing requires running a Triton server on the local machine. Tests are run using a `EfficientNetV2S.tensorflow` model that can be downloaded using pooch:

```python
import pooch
pooch.retrieve(
    fname="EfficientNetV2S.tensorflow.zip",
    url="https://drive.usercontent.google.com/download?id=1Mmm2sRGzdzCEAODjABiiPIiBdg40EPwC&export=download&confirm=t",
    known_hash="b115917b7d0e480fe080077d60913d90c4b596c5a1e3c74745c22f21ed14df63",
    path=host_model_repository
)
```
