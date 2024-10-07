from .data import data, hash_inference, inferred, it_kwargs_icc
import numpy as np
import os
from simple_triton.config import PythonConfig
from simple_triton.feature_extraction import inference
from simple_triton.inference import Requests
from simple_triton.model import TritonModel
from simple_triton.tile_iterators import TiffPrefetch
from .triton import triton


def infer_batch(name, batch, grpc_port):
    req = Requests(f"localhost:{grpc_port}", limit=10)
    request = {
        "model_name": name,
        "inputs": batch,
        "metadata": {"x": 0, "y": 0, "other_stuff": None},
        "times": {},
    }
    req.insert(request, timeout=None)
    completed = req.check(block=True)
    return completed[0]["result"]


def python_batch_compare(data, name, grpc_port, max_batch_size=64, rtol=1e-05, atol=1e-08):
    basic = PythonConfig(name, max_batch_size)
    model = TritonModel(name, f"localhost:{grpc_port}")
    model.load(config=basic.json())
    assert model.is_loaded()
    batch = np.load(data.fetch("batch.npy"))
    truth = np.load(data.fetch(f"{name}_batch.npy"))
    output = infer_batch(name, batch, grpc_port)
    assert np.allclose(truth, output, rtol, atol)
    model.unload(block=True)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def test_inference(data, it_kwargs_icc, inferred, triton):
    _, grpc_port, _ = triton
    iterator = TiffPrefetch(**it_kwargs_icc)
    features, metadata, times, failures = inference(
        iterator, "EfficientNetV2S.tensorflow", url=f"localhost:{grpc_port}"
    )
    result = hash_inference(metadata, np.concatenate(features[0], axis=0))
    assert result.keys() == inferred.keys()
    assert all(
        [cosine_similarity(result[k], inferred[k]) > 0.999 for k in result.keys()]
    )

def verify_huggingface():
    hf_token = os.getenv("HF_TOKEN", "")
    assert hf_token, "No HF_TOKEN environment variable set."

def test_hibou_L(data, triton):
    _, grpc_port, _ = triton
    verify_huggingface()
    python_batch_compare(data, "hibou-L", grpc_port)

def test_phikon(data, triton):
    _, grpc_port, _ = triton
    verify_huggingface()
    python_batch_compare(data, "phikon", grpc_port)

def test_uni(data, triton):
    _, grpc_port, _ = triton
    verify_huggingface()
    python_batch_compare(data, "uni", grpc_port)
