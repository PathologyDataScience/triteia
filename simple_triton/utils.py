import numpy as np
from tabulate import tabulate, SEPARATING_LINE
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


def analyze(times, floatfmt=".2f"):
    """Print a summary table of time spent during inference.

    This table summarizes the total and fractional time spent loading data,
    submitting inference requests, completing requests, and retrieving
    results.

    Parameters
    ----------
    times : dict
        A dictionary returned by `inference` containing read_start,
        read_stop, submitted, retrieved, completed times for each inference.
    floatfmt : str
        A format string for float printing.
    """

    # calculate times
    total = np.subtract(times["retrieved"], times["read_start"])
    read = np.subtract(times["read_stop"], times["read_start"])
    submission = np.subtract(times["submitted"], times["read_stop"])
    completion = np.subtract(times["completed"], times["submitted"])
    retrieval = np.subtract(times["retrieved"], times["completed"])

    # convert to percentages
    read_frac = [100.0 * r / t for (t, r) in zip(total, read)]
    completion_frac = [100.0 * c / t for (t, c) in zip(total, completion)]
    retrieval_frac = [100.0 * r / t for (t, r) in zip(total, retrieval)]
    submission_frac = [100.0 * s / t for (t, s) in zip(total, submission)]

    def stats(x):
        return [np.median(x), np.std(x), min(x), max(x)]

    # form table
    table = [
        ["total (sec)", *stats(total)],
        ["read (sec)", *stats(read)],
        ["submission (sec)", *stats(submission)],
        ["completion (sec)", *stats(completion)],
        ["retrieval (sec)", *stats(retrieval)],
        SEPARATING_LINE,
        ["read (% total)", *stats(read_frac)],
        ["submission (% total)", *stats(submission_frac)],
        ["completion (% total)", *stats(completion_frac)],
        ["retrieval (% total)", *stats(retrieval_frac)],
    ]

    # display results
    print(
        tabulate(table, headers=["", "median", "stdv", "min", "max"], floatfmt=floatfmt)
    )
