from functools import partial
import json
from multiprocessing import get_context
import numpy as np
import pytest
from simple_triton.model import model_config, model_idle, model_metadata, load_model
import sys
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


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


def attend_model(
    event,
    url="localhost:8001",
    model_name="densenet_onnx",
    inputs={"name": "data_0", "datatype": "FP32", "shape": [1, 3, 224, 224]},
    outputs="fc6_1",
    interval=10e-1,
):
    """A testing utility used to submit inference requests to a Triton server.

    This process submits events to a model with period `interval` seconds to
    make the model appear active. This active state is necessary for testing
    functions related to model loading / unloading. Default parameters
    correspond to the `densenet_onnx` model used in Triton examples, served
    from a local instance.

    Parameters
    ----------
    event : multiprocessing.Event
        An event used by the main process to signal the experiment end.
    url : string
        The inference server url. Default value is "localhost:8001".
    model_name : string
        The name of the model to submit to. Default value is "densenet_onnx".
    inputs : dict
        A dictionary defining the name, type, and shape of model inputs.
        Limited to single input models. Default value is for `densenet_onnx`.
    outputs : str
        The name of the model output. Default value is for `densenet_onnx`.
    shape : list or tuple of int
        The shape of each sample to send in [Batch, C, W, H] format.
        Default value is [1, 3, 224, 224].
    interval : float
        Request submission interval in seconds.
    """

    # create client
    client = grpcclient.InferenceServerClient(url=url, verbose=False)

    # define inference callback
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    while True:
        # start timer
        start = time.time()

        # exit if testing process sets event (on experiment end)
        if event.is_set():
            sys.exit(1)

        # build input and output structures for next inference
        # do not reuse since triton output caching for repeated inputs is
        # not understood
        data = np.random.uniform(size=inputs["shape"]).astype(np.float32)
        infer_inputs = [
            grpcclient.InferInput(inputs["name"], inputs["shape"], inputs["datatype"])
        ]
        infer_inputs[0].set_data_from_numpy(data)
        infer_outputs = [grpcclient.InferRequestedOutput(outputs)]

        # infer
        discard = []
        client.async_infer(
            model_name=model_name,
            inputs=infer_inputs,
            callback=partial(callback, discard),
            outputs=infer_outputs,
            client_timeout=None,
        )

        # sleep until next inference
        time.sleep(max(0, interval - (time.time() - start)))


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


def test_model_idle():
    """Check that `model_idle` returns true after a delay. Submit inferences
    with an interval of 0.1 seconds, and after stoping evalaute model_idle
    for various lags to see verify observed transition from busy to idle."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(1.0)

    # end subprocess
    event.set()

    # calculate idle stats at multiple lags
    lags = [
        model_idle(client, MODEL, idle=delta) for delta in np.logspace(-5, 2, num=10)
    ]

    # verify that transition was observed
    assert any(lags) and any([not lag for lag in lags])


def test_load_model_noconfig():
    """Check re-load of model with no config (do nothing)"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # load model with empty config
    load_model(client, MODEL, config=None, verbose=True)

    assert True


def test_load_model_new_config():
    """Check that a currently loaded but idle model can be unloaded/loaded with
    a new config."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # re-use the original config to update - triton doesn't know
    config = json.dumps(model_config(client, MODEL))

    # attempt to unload/load model with retry
    load_model(client, MODEL, config=config, block=True, timeout=2.0, verbose=True)

    assert True


def test_load_model_unloaded_noconfig():
    """Check that an unloaded model can be unloaded/loaded with a config."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # re-use the original config to update - triton doesn't know
    config = json.dumps(model_config(client, MODEL))

    # unload model
    client.unload_model(MODEL)

    # attempt to load model with retry
    load_model(client, MODEL, config=config, block=True, timeout=2.0, verbose=True)

    assert True


def test_load_model_unloaded_config():
    """Check that a currently loaded but idle model can be unloaded/loaded with
    a new config."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # unload model
    client.unload_model(MODEL)

    # attempt to load model with retry
    load_model(client, MODEL, config=None, block=True, timeout=2.0, verbose=True)

    assert True


def test_load_model_busy_retry():
    """Check that a busy model cannot be unloaded/loaded using a retry
    mechanism to handle busy status."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(1.0)

    # provide a trivial config to update
    config = '"parameters": {"config": {{"max_batch_size": "16"}}}'

    # attempt to unload/load model with retry
    try:
        with pytest.raises(InferenceServerException):
            load_model(client, MODEL, config, retries=5, block=False, verbose=True)
    finally:
        # end subprocess
        event.set()


def test_load_model_busy_timeout():
    """Check that a busy model cannot be unloaded/loaded using a retry
    mechanism to handle busy status."""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(1.0)

    # provide a trivial config to update
    config = '"parameters": {"config": {{"max_batch_size": "16"}}}'

    # attempt to unload/load model with retry
    try:
        with pytest.raises(InferenceServerException):
            load_model(client, MODEL, config, block=True, verbose=True)
    finally:
        # end subprocess
        event.set()
