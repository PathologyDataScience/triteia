from functools import partial
import pytest
import subprocess
from time import sleep, time


"""This fixture performs setup and teardown of the triton server container for testing.
The server container is launched using the host docker socket once per test session and
is stopped+removed following tests. Testing will terminate if the container image is
not present on the system. The `triton` fixture receives the model repository from the
data fixture which creates a temporary directory.
"""

TIMEOUT = 10.0  # short timeout for container run, stop operations (not build)
TRITON_IMAGE_NAME = "model-tritonserver"

"""commands to stop, build, and run dockers parameterized by `param`"""
stop_cmd = (
    "docker stop $(docker ps -a -q --filter ancestor={param} --format=\"{{.ID}}\")"
)
exited_cmd = (
    "docker ps -f status=running -f ancestor={param}"
)
build_cmd = (
    "docker build -t model-tritonserver -f {param}/models.Dockerfile ."
)
run_cmd = (
    "docker run --gpus=all -d --rm -p8000:8000 -p8001:8001 -p8002:8002 -p8003:8003 "
    "--shm-size=1g --ulimit memlock=-1 --ipc=host "
    "-v {param}:/models "
    "nvcr.io/nvidia/tritonserver:23.03-py3 "
    "tritonserver --model-repository=/models "
    "--model-control-mode=explicit --exit-on-error=false "
)
cmd = partial(subprocess.run, shell=True, capture_output=True, text=True)


"""Launch the triton container - block until responsive"""


def triton_launch(model_repository):
    if not triton_image_available():
        pytest.exit(
            (
                "tritonserver image not available - aborting tests. "
                "Run docker pull nvcr.io/nvidia/tritonserver:23.03-py3"
            ),
            returncode=2,
        )
    container_id = cmd(run_cmd.format(param=model_repository))
    start = time()
    while not triton_ready() and time() - start < TIMEOUT:
        sleep(1.0)


"""Stop the triton containers - block until stopped"""


def triton_stop(name):
    response = cmd(stop_cmd.format(param=name))
    start = time()
    response = cmd(exited_cmd.format(param=name))
    while len(response.stdout.split("\n")) > 1 and time() - start < TIMEOUT:
        response = cmd(exited_cmd.format(param=name))
        sleep(1.0)


def triton_ready():
    response = cmd('curl -v --silent localhost:8000/v2/health/ready 2>&1 | grep -m 1 "<"')
    return response.stdout.strip() == "< HTTP/1.1 200 OK"


def triton_image_available():
    response = cmd(f"docker images {TRITON_IMAGE_NAME}")
    return TRITON_IMAGE_ID in response.stdout.split("\n")[1]


@pytest.fixture(scope="session")
def triton(model_repository):
    triton_launch(model_repository)
    yield
    closed = container_id = subprocess.run(
        f"docker container stop {container_id.stdout.strip()}",
        shell=True,
        capture_output=True,
        text=True,
    )
    # closed.stdout.strip() == container_id.stdout.strip():
