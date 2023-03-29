from simple_triton.inference import InferenceRunner
import multiprocessing
import multiprocessing.queues
import numpy as np
from tabulate import tabulate
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from model import load_model, model_config, model_update
import json


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


class TimedQueue(multiprocessing.queues.Queue):
    """A queue that records element insertion and removal times."""

    def __init__(self, *args, **kwargs):
        super(TimedQueue, self).__init__(
            *args, **kwargs, ctx=multiprocessing.get_context()
        )

    def put(self, obj, block=True, timeout=None):
        super(TimedQueue, self).put((obj, time.time()), block, timeout)

    def put_nowait(self, obj):
        super(TimedQueue, self).put_nowait((obj, time.time()))

    def get(self, block=True, timeout=None):
        output, insertion = super(TimedQueue, self).get(block, timeout)
        return output, insertion, time.time()

    def get_nowait(self):
        output, insertion = super(TimedQueue, self).get_nowait()
        return output, insertion, time.time()


# analyze and display time performance
def analyze(times, floatfmt=".2f"):
    # calculate times
    total = np.subtract(times["qout_get"], times["qin_put"])
    in_process = np.subtract(times["qout_put"], times["qin_get"])
    qin_time = np.subtract(times["qin_get"], times["qin_put"])
    qout_time = np.subtract(times["qout_get"], times["qout_put"])
    completion = np.subtract(times["completed"], times["submitted"])
    retrieval = np.subtract(times["retrieved"], times["completed"])
    other = [i - (c + r) for (i, c, r) in zip(in_process, completion, retrieval)]

    # convert to percentages
    other = [100.0 * o / i for (i, o) in zip(in_process, other)]
    retrieval = [100.0 * r / i for (i, r) in zip(in_process, retrieval)]
    completion = [100.0 * c / i for (i, c) in zip(in_process, completion)]
    in_process = [100.0 * i / t for (t, i) in zip(total, in_process)]
    qin_time = [100.0 * qi / t for (t, qi) in zip(total, qin_time)]
    qout_time = [100.0 * qo / t for (t, qo) in zip(total, qout_time)]

    # form table
    table = [
        ["total (sec)", np.median(np.array(total)), min(total), max(total)],
        [
            "qin (% total)",
            np.median(np.array(qin_time)),
            min(qin_time),
            max(qin_time),
        ],
        [
            "qout (% total)",
            np.median(np.array(qout_time)),
            min(qout_time),
            max(qout_time),
        ],
        [
            "in-process (% total)",
            np.median(np.array(in_process)),
            min(in_process),
            max(in_process),
        ],
        [
            "completion (% in-process)",
            np.median(np.array(completion)),
            min(completion),
            max(completion),
        ],
        [
            "retrieval (% in-process)",
            np.median(np.array(retrieval)),
            min(retrieval),
            max(retrieval),
        ],
        [
            "other (% in-process)",
            np.median(np.array(other)),
            min(other),
            max(other),
        ],
    ]

    # display results
    print(tabulate(table, headers=["", "median", "min", "max"], floatfmt=floatfmt))


if __name__ == "__main__":
    # server and experiment parameters
    N = 100  # total number of inferences to perform
    limit = 10  # limit on number of pending requests per worker
    workers = 4  # total number of Submitter workers
    url = "localhost:8001"  # url for grpc access to tirton server
    verbose = False  # set verbos as False

    # model configuration parameters
    # input_dtype = "TYPE_FP16"  # set input data type
    # output_dtype = "TYPE_FP32"  # set input data type
    model_name = "simple-trt-model-FP16-test"  # set model name
    batch_size = 1024
    trt = "FP16"
    amp = False
    # dimension_input = 1024
    # dimension_output = 1
    instances = {"count": 4, "kind": "KIND_CPU", "gpus": None}
    optimization = '{"execution_accelerators":{"gpu_execution_accelerator" : [\
           {"name" : "tensorrt", "parameters": {"precision_mode": "FP16"}}]}}'

    # create GRPC client
    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    except InferenceServerException as e:
        print("client creation failed: " + str(e), flush=True)

    # load the test model
    try:
        load_model(client, model_name)
    except InferenceServerException as e:
        print("model loading failed:" + str(e), flush=True)

    # query the model to check the input/output size and type
    config = model_config(client, model_name)
    input_dimension = [int(d) for d in config["input"][0]["dims"]][1:]
    output_dimension = [int(d) for d in config["output"][0]["dims"]]
    input_dtype = config["input"][0]["dataType"]
    output_dtype = config["output"][0]["dataType"]

    # apply an update to the model configuration, changing resources
    config_model_updated = model_update(
        client, model_name, batch_size, instances, trt, amp=False
    )

    # start experiment timer
    start = time.time()

    # create input, output queues
    qin = TimedQueue()
    qout = TimedQueue()

    # Start consumers
    print(f"Creating {workers} workers")
    consumers = [
        InferenceRunner(url, qin, qout, limit, verbose=verbose) for _ in range(workers)
    ]
    for w in consumers:
        w.start()

    # initialize producer
    producer = iter(SimulatedProducer(batch_size, input_dimension, np.float16))

    # enqueue tasks
    print("Enqueuing inference jobs")
    for _ in range(N):
        data = next(producer)
        metadata = {"key": "random stuff"}
        qin.put((model_name, [data], metadata))

    # enqueue stop signals
    for i in range(workers):
        qin.put(None)

    # collect results
    print("Collecting results")
    results = []
    while N:
        output, t_put, t_get = qout.get()
        output["times"]["qout_put"] = t_put
        output["times"]["qout_get"] = t_get
        results.append(output)
        N -= 1
        print(N)

    # display elapsed time
    print(f"Total elapsed time: {time.time()-start}")
    analyze(results)
