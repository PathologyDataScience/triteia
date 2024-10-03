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


def replace_study_path(study, path):
    """replace wsi path within study object"""
    for image in study["slides"].keys():
        study["slides"][image]["filename"] = path
    return study


def hash_metadata(metadata):
    keys = {
        "tile_height",
        "tile_width",
        "tile_top",
        "tile_left",
        "target_magnification",
    }
    return [
        hash_dict({k: metadata[k][i] for k in keys}) 
        for i in range(len(metadata[list(keys)[0]]))
    ]


def cosine_similarity(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


def test_inference(data, triton):
    with open(data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.hash_icc.pkl"), "rb") as f:
        kwargs = pickle.load(f)["kwargs"]
    kwargs["study"] = replace_study_path(kwargs["study"], wsi_path)
    iterator = TiffPrefetch(**kwargs)
    features, metadata, times, failures = inference(
        iterator, "EfficientNetV2S.tensorflow", url="localhost:8001"
    )
    inferred = {
        m:f for m,f in zip(hash_metadata(metadata), np.concatenate(features[0], axis=0))
    }
    with open(data.fetch("TCGA-AN-A0G0-01Z-00-DX1.svs.EfficientNetV2S.tensorflow_224_0_20X.pkl"), "rb") as f:
        contents = pickle.load(f)
    truth = {
        m:f for m,f in zip(hash_metadata(contents["metadata"]), contents["features"])
    }
    assert truth.keys() == inferred.keys()
    assert all([cosine_similarity(truth[k], inferred[k]) > 0.999 for k in truth.keys()])


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

