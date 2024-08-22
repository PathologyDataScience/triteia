# simple-triton

simple-triton is a Python client for inference with the NVIDIA Triton server. It provides model deployment, configuration, and optimization capabilities for the TensorFlow, ONNX, and Python triton backends directly from Python. This was developed to address limitations of the [PyTriton](https://github.com/triton-inference-server/pytriton) package that only supports deployments with the Python backend where TensorRT, XLA, and mixed precision are not available.

# User guide <a name="user-guide"></a>

## Contents

- [Quick start](#quick-start)
    - [Example](#example)
    - [Running the Triton container](#container)
- [Model configuration](#config)
- [Model control](#control)
- [Command-line interfaces](#cli)
    - [Model loading](#cli-loading)
    - [Inference](#cli-inference)
- [Developer guide](#developer-guide)
    - [Testing](#testing)

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
docker run --init --security-opt seccomp:unconfined --network=container:<name of tritonserver docker container> --shm-size=1g -v ${PWD}/test_data:/data:ro --rm --name tritonclient -it simple_triton_client:latest
```
> **_NOTE:_**  `--init` ensures that the docker container has a "master process" to do clean multi-processing. `--network=` lets the docker image see ports from other containers, in this case the triton server. The default shared memory size is now 64MB, so `--shm-size=` is necessary if you are reading large WSIs. `--security-opt seccomp:unconfined` might only be necessary on bigger machines, but it gives your process access to [openblas](https://www.openblas.net/) threads. `--rm` removes the container on exit, beware.

> The client docker utilizes the server docker network so any ports required by the client must be exposed when launching the _server_ container. For example, running a jupyter notebook on the client requires exposing the jupyter port (8888) on the server using -p <your port>:8888.

### Example <a name="example"></a>

The notebook `examples\feature_extraction.ipynb` demonstrates whole-slide image feature extraction. This example requires a running Triton container on the client machine.

### Running the Triton container <a name="container"></a>

simple-triton is tested with [Triton version 23.03](https://github.com/triton-inference-server/server/releases/tag/v2.32.0).

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

### Loading models <a name="cli-loading"></a>
`load_model` is a command line interface for loading models stored in the Triton repository that provides control over batching, response caching, hardware allocation, and optimizations. When loading a model with this CLI the input/output shapes and types are configured by Triton automatically.

Parameters
- `-n` model name
- `-b` backend one of `tensorflow`, `python`, or `onnx`
- `-m` maximum batch size (default 64 samples per batch, 0 for a non-batching model)
- `d` dynamic batching preferred batch size(s) and delay (default disable dynamic batching)
- `r` cache outputs (default outputs are not cached)
- `i` model instances per gpu / cpu (default 1)
- `c` use cpu (default runs inference on GPU)
- `g` number or list of gpus to use (default all GPUs)
- `p` disable pin memory optimization (default enables page-locked memory for model inputs/outputs)
- `a` automatic mixed precision optimization (TensorFlow) (default 32-bit precision)
- `t` TensorRt optimization (TensorFlow, ONNX) (default disable TRT)
- `x` XLA optimization (TensorFlow) one of -1 (disable) 0 (default), 1 (moderate), or 2 (intense) (default is 0 for backend default)
- `u` Unload a model (requires only model name parameter)

Load a model named EfficientNetV2S on the TensorFlow backend
```console
$python load_model.py EfficientNetV2S tensorflow
```

Change the maximum batch size from 64 to 128
```console
$python load_model.py EfficientNetV2S tensorflow -m 64
```

Enable dynamic batching with a preferred batch size of 128 and a max wait of 200 microseconds
```console
$python load_model.py EfficientNetV2S tensorflow -d 128 200
```

Enable the caching of model outputs
```console
$python load_model.py EfficientNetV2S tensorflow -r
```

Load two copies of the model per GPU
```console
$python load_model.py EfficientNetV2S tensorflow -i 2
```

Run the model on CPU instead of GPU
```console
$python load_model.py EfficientNetV2S tensorflow -c
```

Run the model on 4 GPUs
```console
$python load_model.py EfficientNetV2S tensorflow -g 4
```

Run the model on GPUs 0, 2, and 4
```console
$python load_model.py EfficientNetV2S tensorflow -g 0,2,4
```

Disable pinned memory
```console
$python load_model.py EfficientNetV2S tensorflow -p
```

Enable automatic mixed precision for a model hosted on the TensorFlow backend
```console
$python load_model.py EfficientNetV2S tensorflow -a
```

Enable TensorRT optimization for a model hosted on the TensorFlow or ONNX backends. Use half-float precision. Max cached engines (100), minimum segment size (3), and workspace size (4GB) will have default values.
```console
$python load_model.py EfficientNetV2S tensorflow -t
```

Enable XLA optimization for a model hosted on the TensorFlow backend with optimization level 2
```console
$python load_model.py EfficientNetV2S tensorflow -x 2
```

Unload a model with name EfficientNetV2S for any backend
```console
$python load_model.py EfficientNetV2S -u
```

### Inference <a name="cli-inference"></a>
`inference` is a command line interface for model inference with one or more slides and with control of tiling, masking, data loading, and serialization parameters.

Perform inference with the EfficientNetV2S model on a single slide, outputting serialized embeddings to your home directory
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow
```

Optional parameters allow restricting inference to a tissue mask (`-m`)
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -m TCGA-AN-A0G0-01Z-00-DX1.mask.png
```

Store features in float32 precision rather than default float16 (`-f`)
```console
$python feature_extraction.py ~/TCGA-AN-A0G0-01Z-00-DX1.svs ~/ EfficientNetV2S.tensorflow -f
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

## Model configuration <a name="config"></a>
`simple_triton.config` contains model configuration classes that implement backend-specific configuration options. These classes enable configuration of batching behavior, specification of model input/output shapes and types, and backend optimizations. See the Triton documentation on [model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#model-configuration) and [optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) for more details.

Configuration classes like `PythonConfiguration` and `TensorflowConfiguration` take as inputs additional data classes that configure batching and caching behavior, hardware resources, and model input/output signatures.

When Triton is launched with the `--strict-model-config=false` the server will automatically configure basic information like the input/output signatures and the configuration can omit these
```python
from simple_triton.config import TensorflowConfig
from simple_triton.model import TritonModel
name = "mymodel.tensorflow"
config = TensorflowConfig(name, max_batch_size=64)
model = TritonModel(name, "localhost:8001")
model.load(config=config.json())
```

Alternatively, model inputs and output signatures can be defined using the `ModelInput` and `ModelOutput` classes
```python
from simple_triton.config import ModelInput
input = [ModelInput(name="input_0", shape=[224, 224, 3], dtype=np.float32, optional=False)]
config = TensorflowConfig(name, max_batch_size=64, input=input)
```
Variable sized input dimensions can be indicated using a value of -1. 

The `InstanceGroup` class configures the use of CPU or GPU resources GPU resources and the number of model instances hosted on each GPU 
```python
from simple_triton.config import InstanceGroup
instances = InstanceGroup(count=2, kind="gpu", gpus=[0,1,2,3])
config = TensorflowConfig(name=name, instance_group=instances)
```

`TensorflowOptimization` can be used with `TensorflowMixedPrecision`, `TensorflowXla`, and `TensorRt` to activate automatic mixed precision, XLA compilation, or TensorRT optimization 
```python
from simple_triton.config import TensorflowMixedPrecision, TensorflowXla, TensorflowOptimization
amp = TensorflowMixedPrecision()
xla = TensorflowXla(level=2)
optimizer = TensorflowOptimization(amp=amp, xla=xla)
ampxla_config = TensorflowConfig(name=name, max_batch_size=64, optimization=optimization)
```
TensorRt cannot be used concurrently with XLA and mixed precision.

For a non-batching model, set the maximum batch size to zero
```python
config = TensorflowConfig(name, max_batch_size=0)
```

Configuration classes can also save their configuration to a config.pbtxt for automatic file-based configuration

```python
config.save("/model_repository/mymodel.tensorflow/")
```

File-based configuration is useful for distributing models. When configuring and loading models directly in Python, a JSON formatted dictionary is used. Configuration files use the protocol buffer format. Configuration classes handle conversion between these formats.

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

# Inference <a name="inference"></a>

Inference is performed using `inference.inference`. This function consumes data from a simple iterator that emits data/metadata pairs. For single input models, data is provided as a numpy array. For multi-input models, data is provided as a dict of key value pairs linking numpy arrays to model input names (visible from `Model.get_config()`).

`inference` can apply preprocessing functions to data after loading and prior to inference by passing a callable to the `pre` argument. 

> **Note:** For multi-input models all inputs require uniform batch dimensions. Duplicate singleton values where necessary to satisfy this requirement.

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
