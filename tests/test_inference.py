from .data import data
import numpy as np
from simple_triton.config import PythonConfig
from simple_triton.inference import Requests
from simple_triton.model import TritonModel
import subprocess
from .triton import triton


def infer_batch(name, batch):
    req = Requests("localhost:8001", limit=10)
    request = {
        "model_name": name,
        "inputs": batch,
        "metadata": {"x": 0, "y": 0, "other_stuff": None},
        "times": {},
    }
    req.insert(request, timeout=None)
    completed = req.check(block=True)
    return completed[0]["result"]


def python_batch_compare(data, name, max_batch_size=64, rtol=1e-05, atol=1e-08):
    basic = PythonConfig(name, max_batch_size)
    model = TritonModel(name, "localhost:8001")
    model.load(config=basic.json())
    assert model.is_loaded()
    batch = np.load(data.fetch("batch.npy"))
    truth = np.load(data.fetch(f"{name}.npy"))
    output = infer_batch(name, batch)
    assert np.isclose(truth, output, rtol, atol) 
    model.unload(block=True)


def test_hibou_L(data, triton):
    python_batch_compare(data, "hibou-L")


def test_phikon(data, triton):
    python_batch_compare(data, "phikon")


def test_uni(data, triton):
    python_batch_compare(data, "uni")


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

