from google.protobuf.json_format import MessageToDict
import json
import pytest
from simple_triton.config import ConfigBuilder
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


MODEL = "EfficientNetV2S.tensorflow"
URL = "localhost:8001"
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


def test_init():
    """Verify checking of config type as dict"""
    with pytest.raises(ValueError):
        ConfigBuilder(MODEL, config=1)


def test_max_batch_size():
    """Verify max batch size setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    batch = 512
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.max_batch_size(batch)
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["maxBatchSize"] == batch
    with pytest.raises(ValueError):
        builder.max_batch_size(1.5)


def test_response_cache():
    """Verify response cache setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.response_cache(True)
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["responseCache"] == {"enable": True}
    builder.response_cache(False)
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["responseCache"] == {}
    with pytest.raises(ValueError):
        builder.response_cache(1.5)


def test_add_instance_group():
    """Verify response cache setting properly"""

    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.remove_instance_groups()
    assert "instanceGroup" not in builder.config
    builder.add_instance_group(count=2)
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = MessageToDict(client.get_model_config(MODEL))
    assert updated["config"]["instanceGroup"][0]["count"] == 2
    assert updated["config"]["instanceGroup"][0]["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert updated["config"]["instanceGroup"][0]["kind"] == "KIND_GPU"
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = MessageToDict(client.get_model_config(MODEL))
    assert updated["config"]["instanceGroup"][0]["count"] == 1
    assert updated["config"]["instanceGroup"][0]["gpus"] == [0]
    assert updated["config"]["instanceGroup"][0]["kind"] == "KIND_GPU"
    builder.add_instance_group(count=2, kind="cpu")
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = MessageToDict(client.get_model_config(MODEL))
    assert updated["config"]["instanceGroup"][0]["count"] == 1
    assert updated["config"]["instanceGroup"][0]["gpus"] == [0]
    assert updated["config"]["instanceGroup"][0]["kind"] == "KIND_GPU"  
    assert updated["config"]["instanceGroup"][1]["count"] == 2
    assert updated["config"]["instanceGroup"][1]["kind"] == "KIND_CPU"
    with pytest.raises(ValueError):
        builder.add_instance_group(count=1.5)
    with pytest.raises(ValueError):
        builder.add_instance_group(count=1, kind="tpu")
    with pytest.raises(ValueError):
        builder.add_instance_group(count=1, kind="gpu", gpus=0)
    with pytest.raises(ValueError):
        builder.add_instance_group(count=1, kind="gpu", gpus=['0', '1'])


def test_add_mixed_precision():
    """Verify mixed precision setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.add_mixed_precision()
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["optimization"][
        "executionAccelerators"
    ] == {"gpuExecutionAccelerator": [{"name": "auto_mixed_precision"}]}
    builder.remove_mixed_precision()
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert (
        "executionAccelerators" not in MessageToDict(updated)["config"]["optimization"]
    )
    builder = ConfigBuilder(MODEL, client=client)
    builder.add_trt("FP16")
    builder.add_mixed_precision()
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["optimization"][
        "executionAccelerators"
    ] == {"gpuExecutionAccelerator": [{"name": "auto_mixed_precision"}]}


def test_add_trt():
    """Verify mixed precision setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.add_trt("FP16")
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["optimization"][
        "executionAccelerators"
    ] == {
        "gpuExecutionAccelerator": [
            {"name": "tensorrt", "parameters": {"precision_mode": "FP16"}}
        ]
    }
    builder.add_trt("FP32")
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["optimization"][
        "executionAccelerators"
    ] == {
        "gpuExecutionAccelerator": [
            {"name": "tensorrt", "parameters": {"precision_mode": "FP32"}}
        ]
    }
    builder.remove_trt()
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert (
        "executionAccelerators" not in MessageToDict(updated)["config"]["optimization"]
    )
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.add_mixed_precision()
    builder.add_trt("FP16")
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["optimization"][
        "executionAccelerators"
    ] == {
        "gpuExecutionAccelerator": [
            {"name": "tensorrt", "parameters": {"precision_mode": "FP16"}}
        ]
    }
    with pytest.raises(ValueError):
        builder.add_trt("half-float")


def test_add_input():
    """Verify input setting properly."""

    # offline comparison of config
    builder = ConfigBuilder(MODEL, config=BASIC)
    builder.add_input("input_1", "TYPE_FP32", [-1, -1, 3])
    assert builder.config["input"] == [
        {"name": "input_1", "dataType": "TYPE_FP32", "dims": [-1, -1, 3]}
    ]
    builder.add_input("input_1", "TYPE_FP32", [224, 224, 3])
    assert builder.config["input"] == [
        {"name": "input_1", "dataType": "TYPE_FP32", "dims": [224, 224, 3]}
    ]
    builder.add_input("input_2", "TYPE_FP16", [224, 224, 3])
    assert builder.config["input"] == [
        {"name": "input_1", "dataType": "TYPE_FP32", "dims": [224, 224, 3]},
        {"name": "input_2", "dataType": "TYPE_FP16", "dims": [224, 224, 3]},
    ]
    builder.config["input"] = [
        {"name": "input_2", "dataType": "TYPE_FP32", "dims": ["224", "224", "3"]}
    ]
