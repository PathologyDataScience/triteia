from simple_triton.model import model_config, load_model, model_update, instance_group
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

model_name = "simple-trt-model-FP16"  # set model name
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


def check_add_accelerator(accelerator=None):
    if accelerator == "auto_mixed_precision":
        if gpu_accelerator_status(config, accelerator) == False:
            gpu_accelerator_add(config, gpu_accelerator_amp)
    if accelerator == "tensorrt":
        if gpu_accelerator_status(config, "tensorrt") == False:
            gpu_accelerator_add(config, gpu_accelerator_trt("FP16"))
    if accelerator == "tensorrt_FP32":
        if gpu_accelerator_status(config, "tensorrt") == False:
            gpu_accelerator_add(config, gpu_accelerator_trt("FP32"))


def test_update_model_instance_gpu():
    """Evaluate instance group for  kind="gpu","""
    instance = instance_group(model_name, count=1, kind="gpu", gpus=[0])
    assert instance["kind"] == "KIND_GPU"
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_instance_cpu():
    """Evaluate instance group for  kind="cpu","""
    instance = instance_group(model_name, count=1, kind="cpu", gpus=None)
    assert instance["kind"] == "KIND_CPU"
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_amp_False():
    """Evaluate remove amp if amp==False and amp is in current config"""
    # add amp if not already in the config
    check_add_accelerator("auto_mixed_precision")

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") == False
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_trt_None():
    """Evaluate remove trt if trt is None and trt is in current config"""
    # add both trt and amp if not already in the config

    check_add_accelerator("auto_mixed_precision")
    check_add_accelerator("tensorrt")

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == False
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_amp_True_trt_None():
    """Evaluate if amp is True, add only if trt is None"""
    # add trt if not already in the config
    check_add_accelerator("tensorrt")

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") == True
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_trt_NotNone_FP16():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # add amp if not already in the config
    check_add_accelerator("auto_mixed_precision")

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP16", amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == True
    # re-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_trt_NotNone_FP32():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # add amp if not already in the config
    check_add_accelerator("auto_mixed_precision")

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP32", amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == True
    # rre-load the test model to reset the config for next tests
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)
