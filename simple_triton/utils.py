import numpy as np
import tensorflow as tf
from time import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
import pandas as pd
import subprocess
import sys
from tensorboardX import SummaryWriter
import os
from functools import wraps
import psutil
import logging
import tempfile


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

    data = pd.DataFrame(
        data=[
            stats(total),
            stats(read),
            stats(submission),
            stats(completion),
            stats(retrieval),
            stats(read_frac),
            stats(submission_frac),
            stats(completion_frac),
            stats(retrieval_frac),
        ],
        index=[
            "total (sec)",
            "read (sec)",
            "submission (sec)",
            "completion (sec)",
            "retrieval (sec)",
            "read (% total)",
            "submission (% total)",
            "completion (% total)",
            "retrieval (% total)",
        ],
        columns=["median", "stdv", "avg", "min", "max"],
    )

    df = pd.DataFrame(data)

    return df


# poll from the tritonserver using HTTP endpoint
def write_tritonserver_metrics(endpoint, writer, step):
    keys = [
        "nv_inference_count",
        "nv_inference_request_failure",
        "nv_inference_request_success" "nv_inference_request_success",
        "nv_pinned_memory_pool_total_bytes",
        "nv_pinned_memory_pool_used_bytes",
        # Only accessible if you add `--metrics-config summary_latencies=true` to tritonserver
        "nv_inference_request_summary_us_sum",
    ]
    command = 'curl -s "%s" | grep -E "^(%s)" | sed \'s/{.*}//g\'' % (
        endpoint,
        "|".join(keys),
    )
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output = result.stdout.strip().split("\n")
    for line in output:
        if " " in line:
            key, value = line.split(" ", 1)
            writer.add_scalar(key, int(value), step)


def get_username():
    if os.environ.get("USER"):
        return os.environ.get("USER")

    login = None
    # this will typically fail in docker
    try:
        login = os.getlogin()
    except OSError:
        pass

    if not login or not login.isalnum():
        login = "unnamed"

    return login


def init_tb_writer(tb_dir, tb_name, files, extra):
    # 1. get name for tensorboard dst dir. trying to include username since that will ensure
    # that multiple people on one server won't cause write errors
    user = get_username()
    tb_dir = tb_dir or os.path.join(tempfile.gettempdir(), f"tb_{user}")
    tb_name = tb_name or str(time())
    tb_dst = os.path.join(tb_dir, tb_name)

    writer = SummaryWriter(log_dir=tb_dst)
    logging.info(
        f"Writing tensorboard stats to '{tb_dst}' (inspect with `tensoboard --logdir={tb_dst}`)"
    )
    try:
        writer.add_text("git_sha", os.popen("git rev-parse HEAD").read().strip())
    except Exception as e:
        logging.info(
            "could not get git SHA for tensorboard using `git rev-parse HEAD` - this message is safe to ignore"
        )

    writer.add_text("script_name", sys.argv[0])
    writer.add_text("first_filename", files[0][0])
    writer.add_text("last_filename", files[-1][0])

    for key, val in extra.items():
        writer.add_text(key, str(val))

    return writer


def write_analysis_tb(analysis, writer, i):
    writer.add_scalar("read_total_median", analysis["median"]["read (% total)"], i)
    writer.add_scalar(
        "submission_total_median", analysis["median"]["submission (% total)"], i
    )
    writer.add_scalar(
        "retrieval_total_median", analysis["median"]["retrieval (% total)"], i
    )
    writer.add_scalar("read_total_avg", analysis["avg"]["read (% total)"], i)
    writer.add_scalar(
        "submission_total_avg", analysis["avg"]["submission (% total)"], i
    )
    writer.add_scalar("retrieval_total_avg", analysis["avg"]["retrieval (% total)"], i)


def track_method(func, writer, i):
    @wraps(func)
    def wrapper(*args, **kwargs):
        p = psutil.Process()
        initial_disk_io = psutil.disk_io_counters()
        initial_disk_p = p.io_counters()

        start = time()

        result = func(*args, **kwargs)

        elapsed_time = time() - start

        final_disk_io = psutil.disk_io_counters()
        final_disk_p = p.io_counters()
        bytes_read = final_disk_io.read_bytes - initial_disk_io.read_bytes
        chars_read = final_disk_p.read_chars - initial_disk_p.read_chars
        kilobytes_read = int(bytes_read / 1024)
        writer.add_scalar("kilobytes_read", kilobytes_read, i)
        writer.add_scalar("chars_read", chars_read, i)
        writer.add_scalar("kilobytes_read_per_s", kilobytes_read / elapsed_time, i)
        writer.add_scalar("chars_read_per_s", chars_read / elapsed_time, i)
        writer.add_scalar("time_elapsed", elapsed_time, i)

        return result

    return wrapper
