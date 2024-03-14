import numpy as np
from tabulate import tabulate
import tensorflow as tf
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def create_client(url="localhost:8001", verbose=False):
    """Create a grpcclient.

    Parameters
    ----------
    url : string
        The url for the remote-procedure call port of the Triton server.
        Default value is "localhost:8001".
    verbose : bool
        If True the client will emit status messages to stdout. Default
        is False.

    Returns
    -------
    client : grpcclient.InferenceServerClient
        A client
    """

    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    except Exception as e:
        print("context creation failed: " + str(e), flush=True)
    return client


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
            "data loading (% total)",
            np.median(np.array(qin_time)),
            min(qin_time),
            max(qin_time),
        ],
        [
            "results return (% total)",
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
