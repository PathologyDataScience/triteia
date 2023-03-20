# simple-triton

A simple python client for efficient inference with the NVIDIA Triton inference server.

## Installation

This package can be installed using `pip install`. The build and dependencies are defined in pyproject.toml. Editable install is supported.

Using the whole slide image reader requires installation of `histomcs_stream` and `large_image` with tiff and openslide tile sources using a Python wheel

```
sudo apt update
sudo apt install -y python3-openslide openslide-tools
pip install histomics_stream 'large_image[tiff]' \
  scikit_image --find-links https://girder.github.io/large_image_wheels
```

## Example

The example `examples\feature_extraction.ipynb` demonstrates how to extract features from a whole-slide image. This example requires installation of the `mil` library.

## Testing

Testing and code formatting is automated using tox and pytest and can be run using `python -m tox run`. Running this will evaluate the tests in the environments defined in `tox.ini` and will format the source using Black. Following testing, a coverage.html file will be located in .tox/coverage.

Testing requires running a Triton server on the local machine. Tests are run using a `densenet_onnx` model used in the Triton quickstart guide.

### Running Triton server

Download the `densenet_onnx` test model to your host model repository folder:

```
mkdir -p host_model_repository/densenet_onnx/1
wget -O host_model_repository/densenet_onnx/1/model.onnx https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
```

Run Triton as a container and provide your host model repository path to the volume `-v` option:

```
docker run --gpus=8 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 -v/host_model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false --load-model=*
```

The argument `--model-control-mode=explicit` is necessary for the client to load/unload models and to manipulate their configurations while the server is running. The argument `--load-model=*` loads available models from the repository on startup.
