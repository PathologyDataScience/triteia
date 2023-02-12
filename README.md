# simple-triton

A simple python client for efficient inference with the NVIDIA Triton inference server.

### Installation & testing

This package can be installed using `pip install`. The build and dependencies are defined in pyproject.toml.

Testing and code formatting is automated using tox and pytest and can be run using `python -m tox run`. The testing configuration is defined in tox.ini.

### Repository organization

- /simple_triton - source code
- /tests - tests for source
- /benchmarking - scripts and notebooks for generating benchmarking results
- /results - contains outputs of benchmarking experiments

### Running Triton server

Run Triton as a container and provide your host model repository path to the volume `-v` option:

```
docker run --gpus=8 --rm --network host -v/host_model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false
```

The Triton argument `--model-control-mode=explicit` is necessary for the client to load/unload models and to manipulate their configurations.

Triton uses ports 8000-8003. As an alternative to `--network host` these ports can be mapped individually `-p 8000:8000 -p 8001:8001 -p 8002:8002 -p8003:8003`.
