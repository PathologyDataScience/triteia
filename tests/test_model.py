from multiprocessing import get_context
import numpy as np
import pytest
from simple_triton.model import model_config, model_idle, model_metadata, load_model
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from utilities import attend_model


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


def test_model_config_correct():
    """Evalute `model_config` for the testing model and verify that the
    returned dict is correct"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # call model_config on densenet model
    assert model_config(client, MODEL) == CONFIG


def test_model_config_badurl():
    """Evaluate `model_config` when client url is bad"""

    # create client
    client = grpcclient.InferenceServerClient(url="missing.net:8001", verbose=False)

    # call model_config on densenet model
    with pytest.raises(InferenceServerException):
        model_config(client, MODEL)


def test_model_config_badmodel():
    """Evaluate `model_config` when client model name is bad"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # call model_config on densenet model
    with pytest.raises(InferenceServerException):
        model_config(client, "unloaded")


def test_model_metadata_correct():
    """Evalute `model_metadata` for the testing model and verify that the
    returned dict is correct"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # call model_config on densenet model
    assert model_metadata(client, MODEL) == METADATA


def test_model_metadata_badurl():
    """Evaluate `model_metadata` when client url is bad"""

    # create client
    client = grpcclient.InferenceServerClient(url="missing.net:8001", verbose=False)

    # call model_config on densenet model
    with pytest.raises(InferenceServerException):
        model_metadata(client, MODEL)


def test_model_metadata_badmodel():
    """Evaluate `model_metadata` when client model name is bad"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # call model_config on densenet model
    with pytest.raises(InferenceServerException):
        model_metadata(client, "unloaded")


def test_model_idle_false():
    """Check that `model_idle` returns false under timed periodic submissions"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # get fork context for multiprocessing
    context = get_context('fork')

    # for a range of values, make a periodic submissions and check idle
    trials = 10
    factor = 1.1
    for delta in [0.5, 0.1, 0.05]:

        # counter for successful trials
        success = 0

        # create event to signal subprocess experiment end
        event = context.Event()

        # start inference request process
        p = context.Process(target=attend_model, 
                            args=(event,), 
                            kwargs={"interval": delta})
        p.start()
        
        # delay evaluation
        time.sleep(factor*delta)
            
        # conduct trials
        for _ in range(trials):

            # sleep for interval and check model
            time.sleep(delta)
            
            # verify that model is not idle
            success += int(not model_idle(client, MODEL, idle=factor*delta))
            print(success)
    
        # end subprocess
        event.set()
 
        # assert 80% success rate
        assert success / trials >= 0.8


def test_model_idle_true():
    """Check that `model_idle` returns true after a delay"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    
    # check idle status
    context = get_context('fork')

    # create event to signal subprocess experiment end
    event = context.Event()
    
    # start inference request process
    p = context.Process(target=attend_model, 
                        args=(event,), 
                        kwargs={"interval": 0.1})
    p.start()
            
    # delay evaluation to allow inferences to start
    time.sleep(1.)
            
    # end subprocess
    event.set()
            
    # calculate idle stats at multiple lags
    lags = [model_idle(client, MODEL, idle=delta) for delta in np.logspace(-5, 2, num=10)]

    # verify that model was 
    assert any(lags)
