import numpy as np
import tritonclient.grpc as grpcclient
from tabulate import tabulate, SEPARATING_LINE


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
        return client
    except Exception as e:
        print("context creation failed: " + str(e), flush=True)
        raise

def analyze(times, floatfmt=".2f"):
    """Print and return a summary table of time spent during inference as a pandas DataFrame.

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
    
    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the timing statistics.
    """

    # Calculate times
    total = np.subtract(times["retrieved"], times["read_start"])
    read = np.subtract(times["read_stop"], times["read_start"])
    submission = np.subtract(times["submitted"], times["read_stop"])
    completion = np.subtract(times["completed"], times["submitted"])
    retrieval = np.subtract(times["retrieved"], times["completed"])

    # Convert to percentages
    read_frac = [100.0 * r / t for (t, r) in zip(total, read)]
    completion_frac = [100.0 * c / t for (t, c) in zip(total, completion)]
    retrieval_frac = [100.0 * r / t for (t, r) in zip(total, retrieval)]
    submission_frac = [100.0 * s / t for (t, s) in zip(total, submission)]

    def stats(x):
        return [np.median(x), np.std(x), np.mean(x), min(x), max(x)]

    data = pd.DataFrame(data=[stats(total), stats(read), stats(submission), stats(completion), stats(retrieval), stats(read_frac), stats(submission_frac), stats(completion_frac), stats(retrieval_frac)],
        index=["total (sec)", "read (sec)", "submission (sec)", "completion (sec)", "retrieval (sec)", "read (% total)", "submission (% total)", "completion (% total)", "retrieval (% total)"],
        columns=['median', 'stdv', 'avg', 'min', 'max'],
        )

    df = pd.DataFrame(data)

    return df
