from google.protobuf.json_format import MessageToDict
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
            "name": "EfficientNetV2S.tensorflow",
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


def test_max_batch_size():
    """Verify max batch size setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    batch = 512
    builder = ConfigBuilder(config=BASIC)
    builder.max_batch_size(batch)
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["maxBatchSize"] == batch


def test_response_cache():
    """Verify response cache setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(config=BASIC)
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


def test_add_instance_group():
    """Verify response cache setting properly"""

    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(config=BASIC)
    builder.remove_instance_groups()
    assert "instanceGroup" not in builder.config
    builder.add_instance_group(count=2)
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["instanceGroup"] == [
        {
            "count": 2,
            "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
            "kind": "KIND_GPU",
            "name": MODEL,
        }
    ]
    builder.remove_instance_groups()
    builder.add_instance_group(count=1, gpus=[0])
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["instanceGroup"] == [
        {"count": 1, "gpus": [0], "kind": "KIND_GPU", "name": MODEL}
    ]
    builder.add_instance_group(count=2, kind="cpu")
    client.load_model(MODEL, config=json.dumps(builder.config))
    updated = client.get_model_config(MODEL)
    assert MessageToDict(updated)["config"]["instanceGroup"] == [
        {"count": 1, "gpus": [0], "kind": "KIND_GPU", "name": MODEL},
        {"count": 2, "kind": "KIND_CPU", "name": MODEL},
    ]


def test_add_mixed_precision():
    """Verify mixed precision setting properly"""

    # use fast loading of one instance / one gpu
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    builder = ConfigBuilder(config=BASIC)
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
    builder = ConfigBuilder(client=client, model_name=MODEL)
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
    builder = ConfigBuilder(config=BASIC)
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
    builder = ConfigBuilder(config=BASIC)
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


def test_add_input():
    """Verify input setting properly."""

    # offline comparison of config
    builder = ConfigBuilder(config=BASIC)
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
