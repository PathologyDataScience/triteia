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
config =  model_config(client, model_name)

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
    """ Evaluate instance group for  kind="gpu", 
    """
    instance = instance_group(model_name, count=1, kind="gpu", gpus=[ 0 ])
    assert instance['kind'] == 'KIND_GPU'
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_instance_cpu():
    """ Evaluate instance group for  kind="cpu", 
    """
    instance = instance_group(model_name, count=1, kind="cpu", gpus=None)
    assert instance['kind'] == 'KIND_CPU'
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_amp_False():
    """ Evaluate remove amp if amp==False and amp is in current config
    """

    model_update(client, model_name, max_batch_size=None, 
                instances= None, 
                trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") ==False
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_trt_None():
    """ Evaluate remove trt if trt is None and trt is in current config
    """
    model_update(client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") ==False
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


def test_update_model_amp_True_trt_None():
    """ Evaluate if amp is True, add only if trt is None
    """
    model_update(client, model_name, max_batch_size=None, instances=None, trt=None, amp=True
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "auto_mixed_precision") ==True
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

def test_update_model_trt_NotNone():
    """ Evaluate if trt is not none add trt, remove amp if necessary
    """   
    model_update(client, model_name, max_batch_size=None, instances=None, trt="FP16", amp=False
    )
    config = model_config(client, model_name)
    assert gpu_accelerator_status(config, "tensorrt") == True
    # load the test model to reset the config
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)


