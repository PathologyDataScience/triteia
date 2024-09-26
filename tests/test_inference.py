from .data import data
from .triton import triton
import subprocess


def test_triton(data, triton):
    response = subprocess.run(
        'curl -v --silent localhost:8000/v2/health/ready 2>&1 | grep -m 1 "<"',
        shell=True,
        capture_output=True,
        text=True,
    )
    import time
    print("up")
    time.sleep(60)
    assert response.stdout.strip() == "< HTTP/1.1 200 OK"

