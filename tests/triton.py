import pytest
import subprocess
from time import sleep, time


"""This fixture performs setup and teardown of the triton server container for testing.
The server container is launched using the host docker socket once per test session and
is stopped+removed following tests. Testing will terminate if the container image is
not present on the system. The `triton` fixture receives the model repository from the
data fixture which creates a temporary directory.
"""

LAUNCH_TIMEOUT = 10.0  # short timeout for triton launch if image available
TRITON_IMAGE_NAME = "nvcr.io/nvidia/tritonserver"
TRITON_IMAGE_ID = "f6f448fbf332"

"""Host tritonserver run command parameterized by `repository`"""
cmd = (
    "docker run --gpus=all -d --rm -p8000:8000 -p8001:8001 -p8002:8002 -p8003:8003 "
    "--shm-size=1g --ulimit memlock=-1 --ipc=host "
    "-v {repository}:/models "
    "nvcr.io/nvidia/tritonserver:23.03-py3 "
    "tritonserver --model-repository=/models "
    "--model-control-mode=explicit --exit-on-error=false "
)


"""Launch the triton container - block until responsive and stop container on exit"""


def triton_launch(model_repository):
    if not triton_image_available():
        pytest.exit(
            (
                "tritonserver image not available - aborting tests. "
                "Run docker pull nvcr.io/nvidia/tritonserver:23.03-py3"
            ),
            returncode=2,
        )
    container_id = subprocess.run(
        cmd.format(repository=model_repository),
        shell=True,
        capture_output=True,
        text=True,
    )
    start = time()
    while not triton_ready() and time() - start < LAUNCH_TIMEOUT:
        sleep(1.0)


def triton_ready():
    response = subprocess.run(
        'curl -v --silent localhost:8000/v2/health/ready 2>&1 | grep -m 1 "<"',
        shell=True,
        capture_output=True,
        text=True,
    )
    return response.stdout.strip() == "< HTTP/1.1 200 OK"


def triton_image_available():
    response = subprocess.run(
        f"docker images {TRITON_IMAGE_NAME}",
        shell=True,
        capture_output=True,
        text=True,
    )
    return TRITON_IMAGE_ID in result.stdout.split("\n")[1]


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
