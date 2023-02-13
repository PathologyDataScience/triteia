from functools import partial
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


def test_model_idle_false():
    """Check that `model_idle` returns false under timed periodic submissions"""

    # create client
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    # get fork context for multiprocessing
    context = get_context("fork")

    # for a range of values, make a periodic submissions and check idle
    trials = 10
    factor = 1.1
    for delta in [0.5, 0.1, 0.05]:
        # counter for successful trials
        success = 0

        # create event to signal subprocess experiment end
        event = context.Event()

        # start inference request process
        p = context.Process(
            target=attend_model, args=(event,), kwargs={"interval": delta}
        )
        p.start()

        # delay evaluation
        time.sleep(factor * delta)

        # conduct trials
        try:
            for _ in range(trials):
                # sleep for interval and check model
                time.sleep(delta)

                # verify that model is not idle
                success += int(not model_idle(client, MODEL, idle=factor * delta))
        except:
            # end subprocess
            event.set()
            assert False

        # end subprocess
        event.set()

        # assert 80% success rate
        assert success / trials >= 0.8


def test_model_idle_true():
    """Check that `model_idle` returns true after a delay. Submit inferences
    with an interval of 0.1 seconds, and after stoping evalaute model_idle
    for various lags to see transition."""

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
