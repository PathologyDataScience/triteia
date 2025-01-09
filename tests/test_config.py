from google.protobuf.json_format import MessageToDict
import json
import pytest
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
from .triton import triton
from .data import data

from simple_triton.config import InstanceGroup, ModelInput, ModelOutput, TensorflowConfig, TensorflowMixedPrecision, TensorflowOptimization, TensorflowXla, TensorRt
from simple_triton.model import TritonModel


MODEL = "EfficientNetV2S.tensorflow"
CONFIG = {
    "name": "EfficientNetV2S.tensorflow",
    "platform": "tensorflow_savedmodel",
    "versionPolicy": {"latest": {"numVersions": 1}},
    "maxBatchSize": 4,
    "input": [
        {"name": "input_2", "dataType": "TYPE_FP32", "dims": ["224", "224", "3"]}
    ],
    "output": [{"name": "avg_pool", "dataType": "TYPE_FP32", "dims": ["1280"]}],
    "instanceGroup": [
        {
            "count": 1,
            "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
            "kind": "KIND_GPU",
        }
    ],
    "defaultModelFilename": "model.savedmodel",
    "dynamicBatching": {"preferredBatchSize": [4]},
    "optimization": {
        "inputPinnedMemory": {"enable": True},
        "outputPinnedMemory": {"enable": True},
    },
    "backend": "tensorflow",
}
BASIC = {"name": MODEL}


def test_max_batch_size(data, triton):
    """Verify max batch size setting properly"""
    _, grpc_port, _ = triton
    url = f"localhost:{grpc_port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    batch = 512
    model = TritonModel(MODEL, url)

    config = TensorflowConfig(MODEL, max_batch_size=batch)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["maxBatchSize"] == batch

    batch = 256
    config = TensorflowConfig(MODEL, max_batch_size=batch)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["maxBatchSize"] == batch

def test_response_cache(data, triton):
    """Verify response cache setting properly"""
    _, grpc_port, _ = triton
    url = f"localhost:{grpc_port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    model = TritonModel(MODEL, url)

    config = TensorflowConfig(MODEL, max_batch_size=128, response_cache=True)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["responseCache"] == {"enable": True}

    config = TensorflowConfig(MODEL, max_batch_size=128, response_cache=False)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["responseCache"] == {"enable": False} or updated["responseCache"] == {}


def test_add_instance_group(data, triton):
    """Verify setting instance groups properly"""
    _, grpc_port, _ = triton
    url = f"localhost:{grpc_port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    model = TritonModel(MODEL, url)

    instance_group = InstanceGroup(count=0)
    config = TensorflowConfig(MODEL, max_batch_size=128, instance_group=instance_group)
    assert "instanceGroup" in config.json()
    assert not any([x["count"] for x in config.json()["instanceGroup"]])

    instance_group = InstanceGroup(count=2)
    config = TensorflowConfig(MODEL, max_batch_size=128, instance_group=instance_group)
    model.load(config=config.json())
    updated = model.get_config()

    assert updated["instanceGroup"][0]["count"] == 2
    # skipping this test since it's hardware and triton server specific
    # but if anyone want to query triton for available GPUS and check that it returns correct results, feel free..
    # assert updated["instanceGroup"][0]["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert updated["instanceGroup"][0]["kind"] == "KIND_GPU"


    instance_group = InstanceGroup(count=1)
    config = TensorflowConfig(MODEL, max_batch_size=128, instance_group=instance_group)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["instanceGroup"][0]["count"] == 1
    assert updated["instanceGroup"][0]["kind"] == "KIND_GPU"

    instance_group = InstanceGroup(count=1, kind="cpu")
    config = TensorflowConfig(MODEL, max_batch_size=128, instance_group=instance_group)
    model.load(config=config.json())
    updated = model.get_config()
    assert updated["instanceGroup"][0]["count"] == 1
    assert "gpus" not in updated["instanceGroup"][0]

    with pytest.raises(ValueError):
        InstanceGroup(count=1, kind="tpu")
    with pytest.raises(ValueError):
        InstanceGroup(count=1, kind="gpu", gpus=["0"])
    with pytest.raises(ValueError):
        InstanceGroup(count=1, kind="gpu", gpus=["0"])
    with pytest.raises(ValueError):
        InstanceGroup(count=1, kind="gpu", gpus=["0", "1"])


def test_add_mixed_precision(data, triton):
    """Verify mixed precision setting properly"""

    amp = TensorflowMixedPrecision()
    xla = TensorflowXla(level=2)
    optimization = TensorflowOptimization(amp=amp, xla=xla)
    ampxla_config = TensorflowConfig(MODEL, max_batch_size=64, optimization=optimization)

    _, grpc_port, _ = triton
    url = f"localhost:{grpc_port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    model = TritonModel(MODEL, url)

    model.load(config=ampxla_config.json())
    updated = model.get_config()
    assert updated["optimization"][
        "executionAccelerators"
    ] == {"gpuExecutionAccelerator": [{"name": "auto_mixed_precision"}]}


    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    batch = 512
    config = TensorflowConfig(MODEL, max_batch_size=batch)
    model.load(config=config.json())
    updated = model.get_config()
    assert (
        "executionAccelerators" not in updated["optimization"]
    )

    with pytest.raises(ValueError):
        TensorflowOptimization(amp=amp, trt=TensorRt(precision_mode="FP16"))


def test_add_trt(data, triton):
    """Verify trt setting properly"""
    _, grpc_port, _ = triton
    url = f"localhost:{grpc_port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    model = TritonModel(MODEL, url)


    def test_fp(precision_mode):
        trt = TensorRt(precision_mode=precision_mode)
        optimization = TensorflowOptimization(trt=trt)
        config = TensorflowConfig(MODEL, max_batch_size=64, optimization=optimization)
        model.load(config=config.json())
        updated = model.get_config()["optimization"]
        assert "executionAccelerators" in updated and "gpuExecutionAccelerator" in updated["executionAccelerators"]
        updated = updated["executionAccelerators"]['gpuExecutionAccelerator'][0]
        assert "name" in updated and "parameters" in updated and "precision_mode" in updated["parameters"]
        assert updated["name"] == "tensorrt"
        assert updated["parameters"]["precision_mode"] == precision_mode

    test_fp("FP16")
    test_fp("FP32")

    config = TensorflowConfig(MODEL, max_batch_size=64, optimization=None)
    model.load(config=config.json())
    updated = model.get_config()["optimization"]
    assert "optimization" not in updated

    test_fp("FP16")

    with pytest.raises(ValueError):
        trt = TensorRt(precision_mode="half-float")


def test_add_input():
    """Verify input setting properly."""
    inputs = ModelInput(name ="input_1", shape=[-1, -1, 3], dtype=np.float32)
    config = TensorflowConfig(MODEL, max_batch_size=128, input=inputs)

    assert config.json()["input"] == [
        {"name": "input_1", "dataType": "TYPE_FP32", "dims": [-1, -1, 3]}
    ]

    inputs = ModelInput(name ="input_1", shape=[224, 224, 3], dtype=np.float32)
    config = TensorflowConfig(MODEL, max_batch_size=128, input=inputs)
    assert config.json()["input"] == [
        {"name": "input_1", "dataType": "TYPE_FP32", "dims": [224, 224, 3]}
    ]

    inputs = ModelInput(name ="input_2", shape=[224, 224, 3], dtype=np.float32)
    config = TensorflowConfig(MODEL, max_batch_size=128, input=inputs)
    assert config.json()["input"] == [
        {"name": "input_2", "dataType": "TYPE_FP32", "dims": [224, 224, 3]}
    ]


def test_add_output():
    """Verify output setting properly."""
    outputs = ModelOutput(name = "output_1", shape = [1280], dtype=np.float32)
    config = TensorflowConfig(MODEL, max_batch_size=128, output=outputs).json()
    assert "output" in config and len(config["output"]) == 1, config
    assert config["output"][0] == {"name": "output_1", "dataType": "TYPE_FP32", "dims": [1280]}, config["output"][0]

    outputs = ModelOutput(name = "output_2", shape = [4], dtype=np.float16)
    config = TensorflowConfig(MODEL, max_batch_size=128, output=outputs).json()
    assert "output" in config and len(config["output"]) == 1, config
    assert config["output"][0] == {"name": "output_2", "dataType": "TYPE_FP16", "dims": [4]}, config["output"][0]

