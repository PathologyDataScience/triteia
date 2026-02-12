import os
import socket
import subprocess
from contextlib import closing
from functools import partial
from time import sleep, time

import pytest

from .data import data


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


"""This fixture performs setup and teardown of the triton server container for testing.
Running server containers are first stopped and and then the server container is launched
in a `triton` session scope fixture that tears the container down on completion. The `triton` 
fixture receives the model repository path from the data fixture which creates a temporary 
directory.
"""

TIMEOUT = 10.0  # short timeout for container run, stop operations (not build)
TRITON_IMAGE_NAME = "model-tritonserver"
TRITON_DOCKERFILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../server.Dockerfile")
)

"""commands to stop, build, and run dockers parameterized by `param`"""
stop_cmd = 'docker stop $(docker ps -q --filter ancestor={param} --format="{{.ID}}")'
stop_container_cmd = "docker container stop "
running_cmd = "docker ps -f status=running -f ancestor={param}"
build_cmd = f"docker build -t model-tritonserver -f {TRITON_DOCKERFILE} ."
full_path = (
    os.path.join(os.path.dirname(__file__), "..", "models")
    if not os.getenv("TRITON_TMP_DIR")
    else os.getenv("TRITON_TMP_DIR")
)
hf_token = os.getenv("HF_TOKEN", "")
hf_token = f"-e HF_TOKEN={hf_token} " if hf_token else ""
run_cmd = (
    "docker run --gpus=all -d --rm -p {http_port}:8000 -p {grpc_port}:8001 -p {metrics_port}:8002 "
    "--shm-size=1g --ulimit memlock=-1 --ipc=host "
    f"{hf_token}"
    "-v {param}:/models "
    f"-v {full_path}:/models_git "
    f"{TRITON_IMAGE_NAME} -- "
    '/bin/bash -c "rsync -a --ignore-existing /models_git/* /models && tritonserver --model-repository=/models '
    '--model-control-mode=explicit --exit-on-error=false"'
)
cmd = partial(subprocess.run, shell=True, capture_output=True, text=True)


def triton_launch(model_repository):
    """Launch the triton container - block until responsive"""
    http_port, grpc_port, metrics_port = (
        find_free_port(),
        find_free_port(),
        find_free_port(),
    )
    run_str = run_cmd.format(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        param=model_repository,
    )
    run_result = cmd(run_str)
    print(f"triton server launch: {run_str} {run_result.stdout}")
    run_result.check_returncode()
    start = time()
    while not triton_ready(http_port) and time() - start < TIMEOUT:
        sleep(1.0)
    if triton_ready(http_port):
        container_name = run_result.stdout.strip()
        return True, container_name, http_port, grpc_port, metrics_port
    else:
        return False, "", None, None, None


def stop_by_ancestor(name):
    """Stop containers by ancestor name"""
    response = cmd(stop_cmd.format(param=name))
    start = time()
    response = cmd(running_cmd.format(param=name))
    while len(response.stdout.splitlines()) > 1 and time() - start < TIMEOUT:
        sleep(1.0)
        response = cmd(running_cmd.format(param=name))
    return not response.stderr


def stop_by_container(name):
    """Stop containers by instance name"""
    response = cmd(stop_container_cmd + name)
    start = time()
    response = cmd(running_cmd.format(param=name))
    return not response.stderr


def running_by_ancestor(name):
    """Check if containers running by ancestor name"""
    response = cmd(running_cmd.format(param=name))
    return len(response.stdout.splitlines()) == 2


def triton_ready(http_port):
    """Verify that server responds ready"""
    response = cmd(
        f'curl -v --silent localhost:{http_port}/v2/health/ready 2>&1 | grep -m 1 "<"'
    )
    return response.stdout.strip() == "< HTTP/1.1 200 OK"


def triton_image_available():
    """Check that image is available"""
    response = cmd(f"docker images {TRITON_IMAGE_NAME}")
    return len(response.stdout.splitlines()) > 1


@pytest.fixture(scope="session")
def triton(data):
    if running_by_ancestor(TRITON_IMAGE_NAME):
        if not stop_by_ancestor(TRITON_IMAGE_NAME):
            pytest.fail(
                f"Setup: Could not stop triton containers w/ ancestor {TRITON_IMAGE_NAME}"
            )
    if not triton_image_available():
        build_result = cmd(build_cmd)
        build_result.check_returncode()
    ready, container_name, http_port, grpc_port, metrics_port = triton_launch(data.path)
    if ready:
        yield http_port, grpc_port, metrics_port
        if not stop_by_container(container_name):
            pytest.fail(
                f"Teardown: Could not stop triton test container {container_name}."
            )
    else:
        if running_by_ancestor(TRITON_IMAGE_NAME):
            stop_by_ancestor(TRITON_IMAGE_NAME)
        else:
            raise RuntimeError("Error occurred: triton server not ready")
