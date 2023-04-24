from functools import partial
import json
from multiprocessing import get_context
import numpy as np
import pytest
from simple_triton.model import TritonModel
from simple_triton.utils import create_client
import sys
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


MODEL = "EfficientNetV2S.tensorflow"
URL = "localhost:8001"
CONFIG = {
    "name": "EfficientNetV2S.tensorflow",
    "platform": "tensorflow_savedmodel",
    "versionPolicy": {"latest": {"numVersions": 1}},
    "maxBatchSize": 1,
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
    "dynamicBatching": {"preferredBatchSize": [1]},
    "optimization": {
        "inputPinnedMemory": {"enable": True},
        "outputPinnedMemory": {"enable": True},
    },
    "backend": "tensorflow",
}
BASIC = {"name": MODEL}
METADATA = {
    "name": "EfficientNetV2S.tensorflow",
    "versions": ["1"],
    "platform": "tensorflow_savedmodel",
    "inputs": [
        {"name": "input_2", "datatype": "FP32", "shape": ["-1", "224", "224", "3"]}
    ],
    "outputs": [{"name": "avg_pool", "datatype": "FP32", "shape": ["-1", "1280"]}],
}


def attend_model(
    event,
    url="localhost:8001",
    model_name="EfficientNetV2S.tensorflow",
    inputs={"name": "input_2", "datatype": "FP32", "shape": [1, 224, 224, 3]},
    outputs="avg_pool",
    interval=0.1,
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
    client = create_client(url)

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

        # build new input and output structures for each inference
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


def test_get_config_correct():
    """Evalute `get_config` for the testing model and verify that the
    returned dict is correct"""

    model = TritonModel(MODEL, URL)
    if model.is_loaded():
        model.unload()
    model.load(
        config={"maxBatchSize": 1, "dynamicBatching": {"preferredBatchSize": [1]}}
    )
    assert model.get_config() == CONFIG


def test_model_config_badurl():
    """Evaluate `model_config` when client url is bad"""

    with pytest.raises(InferenceServerException):
        model = TritonModel(MODEL, url="missing.net:8001")
        model.get_config()


def test_model_config_badmodel():
    """Evaluate `model_config` when client model name is bad"""

    with pytest.raises(InferenceServerException):
        model = TritonModel("unloaded", URL)
        model.get_config()


def test_model_metadata_correct():
    """Evalute `model_metadata` for the testing model and verify that the
    returned dict is correct"""

    model = TritonModel(MODEL, URL)
    assert model.get_metadata() == METADATA


def test_model_metadata_badurl():
    """Evaluate `model_metadata` when client url is bad"""

    with pytest.raises(InferenceServerException):
        model = TritonModel(MODEL, url="missing.net:8001")
        model.get_metadata()


def test_model_metadata_badmodel():
    """Evaluate `model_metadata` when client model name is bad"""

    with pytest.raises(InferenceServerException):
        model = TritonModel("unloaded", URL)
        model.get_metadata()


def test_model_idle():
    """Check that `model_idle` returns true after a delay. Submit inferences
    with an interval of 0.1 seconds, and after stoping evalaute model_idle
    for various lags to see verify observed transition from busy to idle."""

    # create client
    client = create_client(URL)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(5.0)

    # end subprocess
    event.set()

    # calculate idle stats at multiple lags
    model = TritonModel(MODEL, URL)
    lags = [model.is_idle(idle=delta) for delta in np.logspace(-5, 2, num=10)]

    # verify that transition was observed
    assert any(lags) and any([not lag for lag in lags])


def test_load_model_noconfig():

    model = TritonModel(MODEL, URL)
    model.load()
    assert model.is_loaded()


def test_load_model_new_config():
    """Check that a currently loaded but idle model can be unloaded/loaded with
    a new config."""

    model = TritonModel(MODEL, URL)
    model.load(config={"maxBatchSize": 64}, block=True, timeout=2.0, verbose=True)
    assert model.is_loaded()


def test_load_model_unload():
    """Check that an unloaded model can be unloaded/loaded with a config."""

    model = TritonModel(MODEL, URL)
    model.unload()
    assert not model.is_loaded()


def test_load_model_unloaded_noconfig():
    """Check that an unloaded model can be unloaded/loaded with a config."""

    model = TritonModel(MODEL, URL)
    model.unload()
    model.load(block=True, timeout=2.0, verbose=True)
    assert model.is_loaded()


def test_load_model_unloaded_config():
    """Check that a currently loaded but idle model can be unloaded/loaded with
    a new config."""

    model = TritonModel(MODEL, URL)
    model.unload()
    model.load(config={"maxBatchSize": 256}, block=True, timeout=2.0, verbose=True)
    assert model.is_loaded()


def test_load_model_busy_retry():
    """Check that a busy model cannot be unloaded/loaded using a retry
    mechanism to handle busy status."""

    # create client
    client = create_client(URL)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(5.0)

    # attempt to unload/load model with retry
    model = TritonModel(MODEL, URL)
    try:
        with pytest.raises(InferenceServerException):
            model.load(
                config={"maxBatchSize": 16}, retries=5, block=False, verbose=True
            )
    finally:
        event.set()


def test_load_model_busy_timeout():
    """Check that a busy model cannot be unloaded/loaded using a retry
    mechanism to handle busy status."""

    # create client
    client = create_client(URL)

    # check idle status
    context = get_context("fork")

    # create event to signal subprocess experiment end
    event = context.Event()

    # start inference request process
    p = context.Process(target=attend_model, args=(event,), kwargs={"interval": 0.1})
    p.start()

    # delay evaluation to allow inferences to start
    time.sleep(5.0)

    # attempt to unload/load model with retry
    model = TritonModel(MODEL, URL)
    try:
        with pytest.raises(InferenceServerException):
            model.load(config={"maxBatchSize": 16}, block=True, verbose=True)
    finally:
        event.set()
