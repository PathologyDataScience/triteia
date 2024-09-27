from .data import data
from functools import partial
import os
import pytest
import subprocess
from time import sleep, time


"""This fixture performs setup and teardown of the triton server container for testing.
Running server containers are first stopped and and then the server container is launched
in a `triton` session scope fixture that tears the container down on completion. The `triton` 
fixture receives the model repository path from the data fixture which creates a temporary 
directory.
"""

TIMEOUT = 10.0  # short timeout for container run, stop operations (not build)
TRITON_IMAGE_NAME = "model-tritonserver"
TRITON_DOCKERFILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),"../server.Dockerfile")
)

"""commands to stop, build, and run dockers parameterized by `param`"""
stop_cmd = (
    "docker stop $(docker ps -a -q --filter ancestor={param} --format=\"{{.ID}}\")"
)
running_cmd = (
    "docker ps -f status=running -f ancestor={param}"
)
build_cmd = (
    f"docker build -t model-tritonserver -f {TRITON_DOCKERFILE} ."
)
run_cmd = (
    "docker run --gpus=all -d --rm -p8000:8000 -p8001:8001 -p8002:8002 -p8003:8003 "
    "--shm-size=1g --ulimit memlock=-1 --ipc=host "
    "-v {param}:/models "
    f"{TRITON_IMAGE_NAME} "
    "tritonserver --model-repository=/models "
    "--model-control-mode=explicit --exit-on-error=false "
)
cmd = partial(subprocess.run, shell=True, capture_output=True, text=True)


def triton_launch(model_repository):
    """Launch the triton container - block until responsive"""
    container_id = cmd(run_cmd.format(param=model_repository))
    start = time()
    while not triton_ready() and time() - start < TIMEOUT:
        sleep(1.0)
    if triton_ready():
        return True
    else:
        return False


def stop_by_ancestor(name):
    """Stop containers by ancestor name"""
    response = cmd(stop_cmd.format(param=name))
    start = time()
    response = cmd(running_cmd.format(param=name))
    while len(response.stdout.splitlines()) > 1 and time() - start < TIMEOUT:
        sleep(1.0)
        response = cmd(running_cmd.format(param=name))
    if len(response.stdout.splitlines()) > 1:
        return False
    else:
        return True


def running_by_ancestor(name):
    """Check if containers running by ancestor name"""
    response = cmd(running_cmd.format(param=name))
    return len(response.stdout.splitlines()) == 2


def triton_ready():
    """Verify that server responds ready"""
    response = cmd('curl -v --silent localhost:8000/v2/health/ready 2>&1 | grep -m 1 "<"')
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
    if running_by_ancestor("nvcr.io/nvidia/tritonserver"):
        if not stop_by_ancestor("nvcr.io/nvidia/tritonserver"):
            pytest.fail(
                f"Setup: Could not stop triton containers w/ ancestor nvcr.io/nvidia/tritonserver"
            )
    if not triton_image_available():
        cmd(build_cmd)
    ready = triton_launch(data.path)
    if ready:
        yield
        if not stop_by_ancestor(TRITON_IMAGE_NAME):
            pytest.fail(f"Teardown: Could not stop triton test container.")
    else:
        if running_by_ancestor(TRITON_IMAGE_NAME):
            stop_by_ancestor(TRITON_IMAGE_NAME)

