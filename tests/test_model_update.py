from model import model_config, load_model, model_update, instance_group
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

model_name = "simple-trt-model-FP16"  # set model name
model_name_wrong = "simple-trt-model-FP16-1"
gpu_wrong = "gpu1"
cpu_wrong = "cpu1"
url = "localhost:8001"  # url for grpc access to tirton server
max_batch_size = 1024
verbose = False  # set verbos as False

client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

# load the test model
load_model(client, model_name)
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


def test_update_model_instance_gpu():
    """Evaluate instance group for  kind="gpu","""
    instance = instance_group(model_name, count=1, kind="gpu", gpus=[0])
    assert instance["kind"] == "KIND_GPU"
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_model_instance_gpu_wrong():
    """Evaluate instance group for  kind="gpu","""
    try:
        instance = instance_group(model_name, count=1, kind="gpu", gpus=[0])
        assert instance["kind"] == "KIND_GPU"
        load_model(client, model_name_wrong)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_instance_cpu():
    """Evaluate instance group for  kind="cpu","""
    instance = instance_group(model_name, count=1, kind="cpu", gpus=None)
    assert instance["kind"] == "KIND_CPU"
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)


def test_update_model_instance_cpu_wrong():
    """Evaluate instance group for  kind="cpu","""
    try:
        instance = instance_group(model_name, count=1, kind="cpu", gpus=None)
        assert instance["kind"] == "KIND_CPU"
        load_model(client, model_name_wrong)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_amp_False():
    """Evaluate remove amp if amp==False and amp is in current config"""
    # Before doing this test, add amp if not already in the config
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") == False
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_model_amp_False_wrong():
    """Evaluate remove amp if amp==False and amp is in current config"""
    try:
        # Before doing this test, add amp if not already in the config
        model_update(
            client, model_name_wrong, max_batch_size=None, instances=None, trt=None, amp=True
        )
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_trt_None():
    """Evaluate remove trt if trt is None and trt is in current config"""
    # Before this test, add trt if not already in the config
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP16", amp=False
    )

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == False
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_model_trt_None_wrong():
    """Evaluate remove trt if trt is None and trt is in current config"""
    try:
        # Before this test, add trt if not already in the config
        model_update(
            client, model_name_wrong, max_batch_size=None, instances=None, trt="FP16", amp=False
        )
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_amp_True_trt_None():
    """Evaluate if amp is True, add only if trt is None"""
    # Before this test, make amp False and add trt if not already in the config
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP16", amp=False
    )

    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") == True
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_model_amp_True_trt_None_wrong():
    """Evaluate if amp is True, add only if trt is None"""
    try:   
        # Before this test, make amp False and add trt if not already in the config
        model_update(
            client, model_name_wrong, max_batch_size=None, instances=None, trt="FP16", amp=False
        )
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_trt_NotNone_FP16():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # Before this test, add amp if not already in the config
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP16", amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == True
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)


def test_update_model_trt_NotNone_FP16_wrong():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # Before this test, add amp if not already in the config
    try:
        model_update(
            client, model_name_wrong, max_batch_size=None, instances=None, trt=None, amp=True
        )
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_trt_NotNone_FP32():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # Before this test, add amp if not already in the config
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )
    model_update(
        client, model_name, max_batch_size=None, instances=None, trt="FP32", amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == True
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_model_trt_NotNone_FP32_wrong():
    """Evaluate if trt is not none add trt = "FP16", remove amp if necessary"""
    # Before this test, add amp if not already in the config
    try:
        model_update(
            client, model_name_wrong, max_batch_size=None, instances=None, trt=None, amp=True
        )
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_amp_incorrect_format():
    model_update(
            client, model_name, max_batch_size=None, instances=None, trt=None, amp="Test"
        )
    # re-load the test model to reset the config for next tests
    load_model(client, model_name)

def test_update_trt_incorrect_format():
    model_update(
            client, model_name, max_batch_size=None, instances=None, trt="FP0", amp=None
        )

test_update_model_instance_gpu()
test_update_model_instance_gpu_wrong()


test_update_model_instance_cpu()
test_update_model_instance_cpu_wrong()


test_update_model_amp_False()
test_update_model_amp_False_wrong()

test_update_model_trt_None()
test_update_model_trt_None_wrong()


test_update_model_amp_True_trt_None()
test_update_model_amp_True_trt_None_wrong()


test_update_model_trt_NotNone_FP16()
test_update_model_trt_NotNone_FP16_wrong()


test_update_model_trt_NotNone_FP32()
test_update_model_trt_NotNone_FP32_wrong()


test_update_amp_incorrect_format()

test_update_trt_incorrect_format()

