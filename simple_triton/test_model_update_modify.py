import json
import pytest
from model import model_config, load_model
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

model_name = "simple-trt-model-FP16-test"  # set model name
url = "localhost:8001"  # url for grpc access to tirton server
max_batch_size = 1024
verbose = False  # set verbos as False
try:
    client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
except InferenceServerException as e:
    print("client creation failed: " + str(e), flush=True)

# load the test model
try:
    load_model(client, model_name)
except InferenceServerException as e:
    print("model loading failed:" + str(e), flush=True)
config = model_config(client, model_name)

MODEL = "densenet_onnx"
URL = "localhost:8001"
CONFIG = {
    "name": "densenet_onnx",
    "platform": "onnxruntime_onnx",
    "versionPolicy": {"latest": {"numVersions": 1}},
    "input": [
        {"name": "data_0", "dataType": "TYPE_FP32", "dims": ["1", "3", "224", "224"]}
    ],
    "output": [
        {"name": "fc6_1", "dataType": "TYPE_FP32", "dims": ["1", "1000", "1", "1"]}
    ],
    "instanceGroup": [{"name": "densenet_onnx", "count": 2, "kind": "KIND_CPU"}],
    "defaultModelFilename": "model.onnx",
    "optimization": {
        "inputPinnedMemory": {"enable": True},
        "outputPinnedMemory": {"enable": True},
    },
    "backend": "onnxruntime",
}
METADATA = {
    "name": "densenet_onnx",
    "versions": ["1"],
    "platform": "onnxruntime_onnx",
    "inputs": [
        {"name": "data_0", "datatype": "FP32", "shape": ["1", "3", "224", "224"]}
    ],
    "outputs": [
        {"name": "fc6_1", "datatype": "FP32", "shape": ["1", "1000", "1", "1"]}
    ],
}


def gpu_accelerator_status(config, name):
    # returns True if a gpuExecutionAccelerator with "name" value `name` is present

    if "optimization" in config:
        if "executionAccelerators" in config["optimization"]:
            if (
                "gpuExecutionAccelerator"
                in config["optimization"]["executionAccelerators"]
            ):
                for d in config["optimization"]["executionAccelerators"][
                    "gpuExecutionAccelerator"
                ]:
                    if "name" in d:
                        if d["name"] == name:
                            return True
                return False
            else:
                return False
        else:
            return False
    else:
        return False


def gpu_accelerator_delete(config, name):
    # delete a gpuExecutionAccelerator and cleanup empty parents

    if gpu_accelerator_status(config, name):
        gpuexecacc = config["optimization"]["executionAccelerators"][
            "gpuExecutionAccelerator"
        ]
        keep = []
        for i, d in enumerate(gpuexecacc):
            if "name" in enumerate(d):
                if d["name"] != name:
                    keep.append(i)
        config["optimization"]["executionAccelerators"]["gpuExecutionAccelerator"] = [
            gpuexecacc[i] for i in keep
        ]
        if (
            len(
                config["optimization"]["executionAccelerators"][
                    "gpuExecutionAccelerator"
                ]
            )
            == 0
        ):
            del config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ]
        if config["optimization"]["executionAccelerators"] == {}:
            del config["optimization"]["executionAccelerators"]
        if config["optimization"] == {}:
            del config["optimization"]


def gpu_accelerator_trt(precision="FP16"):
    # returns a trt configuration
    if precision.upper() == "FP16" or precision.upper() == "FP32":
        return {
            "name": "tensorrt",
            "parameters": {"precision_mode": f"{precision.upper()}"},
        }
    else:
        raise ValueError("precision must be one of None, numpy.float16, numpy.float32")


def gpu_accelerator_amp():
    # returns an amp configuration
    return {"name": "auto_mixed_precision"}


def gpu_accelerator_add(config, accelerator=gpu_accelerator_trt("FP16")):
    # adds a gpu accelerator to a config
    if gpu_accelerator_status(config, accelerator["name"]):
        gpu_accelerator_delete(config, accelerator["name"])
    if "optimization" not in config:
        config["optimization"] = {}
    if "executionAccelerators" not in config["optimization"]:
        config["optimization"]["executionAccelerators"] = {}
    if "gpuExecutionAccelerator" not in config["optimization"]["executionAccelerators"]:
        config["optimization"]["executionAccelerators"]["gpuExecutionAccelerator"] = []
    config["optimization"]["executionAccelerators"]["gpuExecutionAccelerator"].append(
        accelerator
    )


def instance_group(model_name, count, kind="gpu", gpus=None):
    """Generates an instance count dictionary for use in a model config.

    A Triton configuration allows specification of resources used to serve a
    model. The instance group specifies the number of concurrent instances of
    a model to serve for a given set of resources. Resources can specify cpu
    or gpu hosting, or specific gpus. See Triton documentation for more
    details.

    Parameters
    ----------
    count : int
        The number of model instances to run concurrently.
    kind : str
        One of {"cpu", "gpu"}. Default value of "gpu" specifies that `count`
        models be hosted on each available gpu.
    gpus : list of int
        If specified, `count` instances will be hosted on each of the listed
        gpus. For example, [0, 1] would specify serving on gpus zero and one.
        Default value of `None` means that `count` instances will be served on
        each available gpu.
    """

    if not isinstance(count, int):
        # raise ValueError("argument 'count' must be int.")
        print()
    if kind.lower() not in ["cpu", "gpu"]:
        raise ValueError("argument 'kind' must be one of 'cpu', 'gpu'.")
    elif kind == "cpu":
        kind = "KIND_CPU"
    else:
        kind = "KIND_GPU"
    if gpus is not None:
        if not isinstance(gpus, list):
            raise ValueError("argument 'gpus' must be list of int.")
        for index in gpus:
            if not isinstance(index, int):
                raise ValueError("elements of 'gpus' must be int.")

    # set model name, count, and kind
    instance = {"name": model_name, "count": count, "kind": kind}
    if gpus is not None:
        instance["gpus"] = gpus

    return instance


def instance_group_add(config, instance):
    config["instance_group"].append(instance)


def test_update_model_batch_size():
    """Evaluate batch dimensions is set"""
    config = model_config(client, model_name)
    batch_dim = [input["dims"][0] == "-1" for input in config["input"]]
    if all(batch_dim):
        config["maxBatchSize"] = str(max_batch_size)
    load_model(client, model_name, config=json.dumps(config), block=True)
    config = model_config(client, model_name)
    assert config["maxBatchSize"] == max_batch_size


def test_update_model_instance_gpu(config):
    """Evaluate instance group for  kind="gpu",
    add instance to config,
    re-check config for updates instance"""
    count = 1
    kind = "gpu"
    gpus = "[ 0 ]"
    instance = instance_group(model_name, count, kind, gpus)
    instance_group_add(config, instance)
    load_model(client, model_name, config=json.dumps(config), block=True)
    config = model_config(client, model_name)

    for d in config["instance"]:
        if "kind" in d:
            assert d[kind] == "KIND_GPU"


def test_update_model_instance_cpu():
    """Evaluate instance group for  kind="cpu","""
    count = 1
    kind = "cpu"
    gpus = None
    instance = instance_group(model_name, count, kind, gpus)
    instance_group_add(config, instance)
    load_model(client, model_name, config=json.dumps(config), block=True)
    config = model_config(client, model_name)

    for d in config["instance"]:
        if "kind" in d:
            assert d[kind] == "KIND_CPU"


def test_update_model_Not_amp(config):
    """Evaluate remove amp if amp==False and amp is in current config"""
    amp = False
    # add amp if not exist in current config for test scenario
    if gpu_accelerator_status(config, "auto_mixed_precision") == False:
        gpu_accelerator_add(config, gpu_accelerator_amp)

    if gpu_accelerator_status(config, "auto_mixed_precision"):
        gpu_accelerator_delete(config, "auto_mixed_precision")

    # apply model update
    load_model(client, model_name, config=json.dumps(config), block=True)

    # re-check the model to see if the update was  successful
    # acquire the current model config
    config = model_config(client, model_name)

    # check if amp test is passed
    with pytest.raises(Exception):
        assert gpu_accelerator_status(config, "tensorrt") == True


def test_update_model_trt_None(config):
    """Evaluate remove trt if trt is None and trt is in current config"""
    trt = None
    # add trt if not exist only for test scenario
    if gpu_accelerator_status(config, "tensorrt") == False:
        gpu_accelerator_add(config, gpu_accelerator_trt("FP16"))

    if gpu_accelerator_status(config, "tensorrt"):
        gpu_accelerator_delete(config, "tensorrt")

    # apply model update
    load_model(client, model_name, config=json.dumps(config), block=True)

    config = model_config(client, model_name)

    # check if amp test is passed
    with pytest.raises(Exception):
        assert gpu_accelerator_status(config, "tensorrt") == False


def test_update_model_amp_trt_None(config):
    """Evaluate when amp is True, add only if trt is None"""
    amp = True
    trt = None
    # remove trt since both amp and trt cannot exist together
    # suggested to add this check in model.py?
    if gpu_accelerator_status(config, "tensorrt"):
        gpu_accelerator_delete(config, "tensorrt")

    gpu_accelerator_add(config, gpu_accelerator_amp())

    # apply model update
    load_model(client, model_name, config=json.dumps(config), block=True)

    config = model_config(client, model_name)

    # check if amp test is passed
    with pytest.raises(Exception):
        assert gpu_accelerator_status(config, "auto_mixed_precision") == True


def test_update_modeL_trt_Not_None(config):
    """if trt is not none add trt, remove amp if necessary"""
    trt = "FP16"

    if gpu_accelerator_status(config, "auto_mixed_precision"):
        gpu_accelerator_delete(config, "auto_mixed_precision")
    gpu_accelerator_add(config, gpu_accelerator_trt(trt))

    # apply model update
    load_model(client, model_name, config=json.dumps(config), block=True)

    config = model_config(client, model_name)

    # check if trt existß
    with pytest.raises(Exception):
        assert gpu_accelerator_status(config, "tensorrt") == True


test_update_model_batch_size()
