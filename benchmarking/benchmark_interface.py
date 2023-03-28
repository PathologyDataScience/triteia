import argparse
import numpy as np
import time
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Define the model and server parameters
MODEL_NAME = "densenet_onnx"
MODEL_VERSION = -1  # use the latest version
BATCH_SIZE = 1024
PROTOCOL = "grpc"
TRITON_SERVER_URL = "localhost:8001"

# Define the experiment parameters
N = 100
LIMIT = 10
WORKERS = 4

class SimulatedProducer(object):
    """A simulated producer that emits numpy arrays with specified batch size
    and feature dimensions.

    Data is uniformly distributed and so compression ratio will be low.
    """

    def __init__(self, B=1024, D=[1024], dtype=np.float16):
        """Constructor.

        Parameters
        ----------
        B : int
            Batch size. Default value is 1024.
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



def infer(client, inputs):
    """
    Run a single inference request on the Triton Inference Server.

    Args:
        client (tritonclient.grpc.InferenceServerClient):
        inputs (dict): The inputs to the model.

    Returns:
        The outputs of the model.
    """
    inputs = [grpcclient.InferInput(name, shape, datatype) for name, (shape, datatype) in inputs.items()]
    for input_ in inputs:
        input_.set_data_from_numpy(next(SimulatedProducer()))

    outputs = [grpcclient.InferRequestedOutput(name) for name in client.get_model_metadata(MODEL_NAME).outputs]
    result = client.infer(MODEL_NAME, inputs, outputs=outputs)

    return {output.name: output.as_numpy() for output in result.as_numpy_iterator()}


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


def measure_throughput():
    """
    Measure the throughput of the Triton Inference Server with the DenseNet ONNX model using TensorRT, auto mixed
    precision, GPU, CPU, batch size of 1024, and gRPC protocol. Measure the throughput while enqueuing the inference
    jobs using multiprocessing and multi-workers.
    """
    # Initialize the client for communicating with the Triton Inference Server
    if PROTOCOL == "grpc":
        client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
    else:
        raise ValueError(f"Unsupported protocol: {PROTOCOL}")

    # Warm up the server by running a single inference request
    inputs = {
        "input": ((BATCH_SIZE, 3, 224, 224), "FP16")
    }
    infer(client, inputs)

    # Create the queue of pending inference requests
    queue = Queue(maxsize=LIMIT)
    for i in range(N):
        queue.put({
            "input": ((BATCH_SIZE, 3, 224, 224), "FP16")
        })

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