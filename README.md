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

Generate your own model using `feature_extraction.feature_extractor()` or download the `EfficientNetV2S.tensorflow` test model archive to your host model repository folder:

```python
import pooch
pooch.retrieve(
    fname="EfficientNetV2S.tensorflow.zip",
    url="https://drive.google.com/uc?export=download&id=1Mmm2sRGzdzCEAODjABiiPIiBdg40EPwC&confirm=t&uuid=b11e409a-64b2-4146-b45d-4f229093cb5a&at=ANzk5s7UvBzB7zpqm7AvngovJwS8:1681783040828",
    known_hash="a6ed53d8343498b4ebfe7ff1a9ccbcabef23d6a164d2a521916774af49996f7e",
    path=host_model_repository
)
```

then unzip the archive and remove the original file

```
> unzip host_model_repository/EfficientNetV2S.tensorflow.zip
> rm host_model_repository/EfficientNetV2S.tensorflow.zip
```

Run Triton as a container and provide your host model repository path to the volume `-v` option:

```
docker run --gpus=8 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 -v host_model_repository:/models nvcr.io/nvidia/tritonserver:22.05-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false --strict-model-config=false --load-model=*
```

The argument `--model-control-mode=explicit` is necessary for the client to load/unload models and to manipulate their configurations while the server is running. The argument `--strict-model-config=false` allows Triton to automatically generate model configurations for models found in the repository. The argument `--load-model=*` loads available models from the repository on startup.
