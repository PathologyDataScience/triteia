# simple-triton

A simple python client for efficient inference with the NVIDIA Triton inference server.

### Installation & testing

This package can be installed using `pip install`. The build and dependencies are defined in pyproject.toml. Editable install is supported.

Testing and code formatting is automated using tox and pytest and can be run using `python -m tox run`. Running this will evaluate the tests in the environments defined in `tox.ini` and will format the source using Black.

### Repository organization

- /simple_triton - source code
- /tests - tests for source
- /benchmarking - scripts and notebooks for generating benchmarking results
- /results - contains outputs of benchmarking experiments

### Running Triton server

To download a test model to your host model repository folder:

```
mkdir -p host_model_repository/densenet_onnx/1
wget -O host_model_repository/densenet_onnx/1/model.onnx https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
```

Run Triton as a container and provide your host model repository path to the volume `-v` option:

```
docker run --gpus=8 --rm --network host -v/host_model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false --load-model=*
```

The argument `--model-control-mode=explicit` is necessary for the client to load/unload models and to manipulate their configurations while the server is running. The argument `--load-model=*` loads available models from the repository on startup.

Triton uses ports 8000-8003. The above command uses `--network host` to share the host network with the container. If more isolation is desired, these ports can be mapped individually `-p 8000:8000 -p 8001:8001 -p 8002:8002 -p8003:8003`.
