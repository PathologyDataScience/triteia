import sys

sys.path.append("tritonClient/triton_testing_5/simple_triton")
import numpy as np
import time
import tritonclient.grpc as grpcclient
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from simple_triton.model import model_config, model_metadata
from tritonclient.utils import InferenceServerException
from functools import partial

# Define the model and server parameters
MODEL_NAME = "densenet_onnx"
MODEL_VERSION = -1  # use the latest version
BATCH_SIZE = 2048
PROTOCOL = "grpc"
TRITON_SERVER_URL = "localhost:8001"

# Define the experiment parameters
N = 10
LIMIT = 10
WORKERS = 4


class SimulatedProducer(object):
    """A simulated producer that emits numpy arrays with specified batch size
    and feature dimensions.

    Data is uniformly distributed and so compression ratio will be low.
    """

    def __init__(self, B=2048, D=[1024], dtype=np.float16):
        """Constructor.

        Parameters
        ----------
        B : int
            Batch size. Default value is 2048.
        D : list of int
            Feature dimensions. Default value is [1024].
        dtype : numpy.dtype
            A numpy dtype for the emited data. Default value is float16.
        """

        self.B = B  # batch size
        self.D = D  # dimensions
        self.dtype = dtype  # datatype as float16 or float32

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        output = self.dtype(np.random.uniform(size=(self.B, *self.D)))
        return output


def _callback(self, capture, result, error):
    """Callback for async_infer to capture result or error of inference
    request.

    Parameters
    ----------
    capture : list
        An empty list of
    result : grpcclient.InferResult
        The result of inference if successful.
    error : tritonclientutils.InferenceServerException
        An exception if inference failed. Otherwise None.
    """

    if error:
        capture.append((error, time.time()))
    else:
        capture.append((result, time.time()))


def _client_outputs(self, model_dict):
    """Generates tritonclient.grpc.InferRequestedOutput objects for client.

    Parameters
    ----------
    model_dict : dict
        A dictionary describing the name, shape, and type of inputs and
        outputs, as well as maximum batch size.

    Outputs
    -------
    infer_outputs : list of tritonclient.grpc.InferRequestedOutput
        A list of InferRequestedOutput objects to capture inference
        results.
    """

    import tritonclient.grpc as grpcclient

    # create InferInput objects for each model input
    infer_outputs = [
        grpcclient.InferRequestedOutput(o["name"]) for o in model_dict["outputs"]
    ]

    return infer_outputs


def infer(client, sample, timeout=None):
    """
    Perform inference on the input using the Triton Inference Server.

    Args:
        client (tritonclient.grpc.InferenceServerClient): The client used to communicate with the Triton Inference Server.
        inputs (dict): A dictionary containing the input tensor.

    Returns:
        The output tensor as a numpy array.
    """
    try:

        model_dict = {
            **model_metadata(client, model_name),
            "max_batch_size": model_config(client, model_name)["maxBatchSize"],
        }
        model_dicts[model_name] = model_dict
        # create InputData objects based on data shape
        inputs = _client_inputs(sample["inputs"], model_dict)

        # create outputs
        outputs = _client_outputs(model_dict)

        # initialize output
        sample["result"] = []

        # increment attempts
        if "attempts" not in sample.keys():
            sample["attempts"] = 0
        sample["attempts"] = sample["attempts"] + 1

        # submit request

        response = client.async_infer(
            model_name=model_name,
            inputs=inputs,
            callback=partial(_callback, sample["result"]),
            outputs=outputs,
            client_timeout=timeout,
        )
        output = response.as_numpy("output")
        return output
    except InferenceServerException as e:
        print(f"Inference failed: {repr(e)}")
        return None


def _client_inputs(self, inputs, model_dict):
    """Generates tritonclient.grpc.InferInput objects for client.

    Given inputs and a model configuration and metadata, this function
    validates the inputs against the model expectations, and generates the
    InferInput objects used by the client to transmit inputs to triton.

    Parameters
    ----------
    inputs : list of numpy.ndarray
        A list of numpy arrays to input for model inference.
    model_dict : dict
        A dictionary describing the name, shape, and type of inputs and
        outputs, as well as maximum batch size.

    Outputs
    -------
    infer_inputs : list of tritonclient.grpc.InferInput
        A list of InferInput objects populated with data.
    """

    import tritonclient.grpc as grpcclient

    # validate inputs against model expectations
    self._validate_inputs(inputs, model_dict)

    # create InferInput objects for each model input
    infer_inputs = []
    for provided, expected in zip(inputs, model_dict["inputs"]):
        # create InferInput object
        iio = grpcclient.InferInput(
            expected["name"], provided.shape, expected["datatype"]
        )

        # add numpy data to input
        if self._np_to_api_types(provided.dtype) != expected["datatype"]:
            provided = provided.astype(self._api_to_np_types(expected["datatype"]))
        iio.set_data_from_numpy(provided)

        # add to infer_input list
        infer_inputs.append(iio)

    return infer_inputs


def submitter(worker_id, queue, client):
    """
    Worker function for submitting inference requests to the Server.

    Args:
        worker_id (int): The ID of the worker.
        queue (queue.Queue): The queue of pending inference requests.
    """
    while True:
        batch = []
        while len(batch) < BATCH_SIZE:
            try:
                inputs = queue.get(timeout=1)
                batch.append(inputs)
            except:
                break

        if not batch:
            break

        results = []
        for inputs in batch:
            results.append(infer(client, inputs))
        queue.task_done()


def measure_throughput(model_dicts):
    """
    Measure the throughput of the Triton Inference Server with the DenseNet ONNX model using TensorRT, auto mixed
    precision, GPU, CPU, batch size of 2048, and gRPC protocol. Measure the throughput while enqueuing the inference
    jobs using multiprocessing and multi-workers.
    """
    # Initialize the client for communicating with the Triton Inference Server
    if PROTOCOL == "grpc":
        client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
    else:
        raise ValueError(f"Unsupported protocol: {PROTOCOL}")

    # Warm up the server by running a single inference request
    inputs = {"input": (input_dimension, np.float16)}

    infer(client, inputs)

    # Create the queue of pending inference requests
    queue = Queue(maxsize=LIMIT)
    producer = iter(SimulatedProducer(BATCH_SIZE, input_dimension, np.float16))
    for i in range(N):

        data = next(producer)
        metadata = {"key": "random stuff"}
        queue.put((model_name, [data], metadata))

    # Start the workers to submit inference requests to the server
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = []
        for worker_id in range(WORKERS):
            futures.append(executor.submit(submitter, worker_id, queue, client))

        start_time = time.time()

        # Wait for all the inference requests to complete
        for future in futures:
            future.result()

        end_time = time.time()

    # Calculate the throughput in inferences per second
    throughput = N / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} inferences/sec")


if __name__ == "__main__":
    url = "localhost:8001"  # url for grpc access to tirton server
    verbose = False  # set verbos as False
    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    except InferenceServerException as e:
        print("client creation failed: " + str(e), flush=True)
        # load the test model

    model_name = "simple-trt-model-FP16"  # set model name
    # query the model to check the input/output size and type
    config = model_config(client, model_name)
    input_dimension = [int(d) for d in config["input"][0]["dims"]][0:]
    output_dimension = [int(d) for d in config["output"][0]["dims"]]
    input_dtype = config["input"][0]["dataType"]
    output_dtype = config["output"][0]["dataType"]
    model_dicts = {}
    measure_throughput(model_dicts)
