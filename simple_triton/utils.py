import threading
from datetime import timedelta

import time
import tritonclient.grpc as grpcclient
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
    retrieval = np.subtract(times["retrieved"], times["submitted"])

    # Convert to percentages
    read_frac = [100.0 * r / t for (t, r) in zip(total, read)]
    retrieval_frac = [100.0 * r / t for (t, r) in zip(total, retrieval)]
    submission_frac = [100.0 * s / t for (t, s) in zip(total, submission)]

    def stats(x):
        return [np.median(x), np.std(x), np.mean(x), min(x), max(x)]

    data = pd.DataFrame(
        data=[
            stats(total),
            stats(read),
            stats(submission),
            stats(retrieval),
            stats(read_frac),
            stats(submission_frac),
            stats(retrieval_frac),
        ],
        index=[
            "total (sec)",
            "read (sec)",
            "submission (sec)",
            "retrieval (sec)",
            "read (% total)",
            "submission (% total)",
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
    tb_name = tb_name or str(time.time())
    tb_dst = os.path.join(tb_dir, tb_name)

    writer = SummaryWriter(log_dir=tb_dst)
    logging.info(
        f"Writing tensorboard stats to '{tb_dst}' (inspect with `tensorboard --logdir={tb_dst}`)"
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


def monitor_gpu(stop_event, interval_sec, action, writer, gpu_index=0):
    import importlib.util

    pynvml_module = importlib.util.find_spec("pynvml")
    if pynvml_module is not None:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetUtilizationRates,
        )
    else:
        return
    """
    Periodically logs GPU memory and processor usage.

    Args:
        stop_event: A threading.Event that signals the monitor to stop.
        interval: Time in seconds between each log.
        action: A string identifier for the action being monitored.
        writer: A logging writer (e.g., TensorBoard writer).
        gpu_index: Index of the GPU to monitor (default: 0).
    """
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        i = 0
        while not stop_event.is_set():
            # Memory usage
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_used = mem_info.used / (1024 * 1024)  # Convert to MB
            gpu_mem_total = mem_info.total / (1024 * 1024)  # Convert to MB

            # GPU utilization
            utilization = nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu  # GPU utilization percentage

            # Log metrics
            writer.add_scalar(
                f"{action}_gpu_{gpu_index}_mem_used_mb", gpu_mem_used, global_step=i
            )
            writer.add_scalar(
                f"{action}_gpu_mem_{gpu_index}_total_mb", gpu_mem_total, global_step=i
            )
            writer.add_scalar(
                f"{action}_gpu_{gpu_index}_util_percent", gpu_util, global_step=i
            )
            i += 1

            time.sleep(interval_sec)
    finally:
        nvmlShutdown()


def monitor_memory(stop_event, interval, action, writer):
    """Periodically logs memory usage of the current process."""
    process = psutil.Process(os.getpid())
    i = 0
    while not stop_event.is_set():
        mem_info = process.memory_info()
        mem_used = mem_info.rss / (1024 * 1024)
        writer.add_scalar(f"{action}_mem_mb", mem_used, global_step=i)
        i += 1
        time.sleep(interval)


def monitor_cpu(stop_event, interval, action, writer, python_only=False):
    """Periodically logs CPU usage.

    - Default: logs one scalar per CPU on the host system.
    - If python_only=True: logs CPU usage for Python processes (total and per-PID).
    """
    process = psutil.Process(os.getpid())
    i = 0

    # Prime psutil CPU measurements to avoid 0.0 on first read
    psutil.cpu_percent(percpu=True, interval=None)
    process.cpu_percent(interval=None)

    while not stop_event.is_set():
        if python_only:
            # Collect CPU percent for all Python processes
            total_py_cpu = 0.0
            per_pid = {}
            for p in psutil.process_iter(attrs=["pid", "name"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if "python" in name:
                        # cpu_percent(None) returns the percent since last call
                        per = p.cpu_percent(interval=None)
                        per_pid[p.info["pid"]] = per
                        total_py_cpu += per
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            writer.add_scalar(
                f"{action}_cpu_percent_python_total", total_py_cpu, global_step=i
            )
            # Also log per-PID to help disambiguate multiple Python workers
            for pid, per in per_pid.items():
                writer.add_scalar(
                    f"{action}_cpu_percent_python_pid_{pid}", per, global_step=i
                )
        else:
            # System-wide per-CPU usage
            per_cpu = psutil.cpu_percent(percpu=True, interval=None)
            writer.add_scalar(
                f"{action}_cpu_percent_total/avg",
                sum(per_cpu) / len(per_cpu),
                global_step=i,
            )
            writer.add_scalar(
                f"{action}_cpu_percent_total/agg", sum(per_cpu), global_step=i
            )
            for idx, val in enumerate(per_cpu):
                writer.add_scalar(f"{action}_cpu_percent/cpu_{idx}", val, global_step=i)

        i += 1
        time.sleep(interval)


def monitor_disk(stop_event, interval, action, writer, path="/"):
    """Periodically logs disk usage percentage for the specified path."""
    i = 0
    while not stop_event.is_set():
        disk_usage = psutil.disk_usage(path)
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        writer.add_scalar(f"{action}_disk_usage_percent", disk_percent, global_step=i)
        i += 1
        time.sleep(interval)


def track_method(func, writer, slide_num, live_tracking=False, path="/"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        p = psutil.Process()
        initial_disk_io = psutil.disk_io_counters()
        initial_disk_p = p.io_counters()

        if live_tracking:
            stop_event = threading.Event()
            memory_tracker = threading.Thread(
                target=monitor_memory,
                args=(stop_event, 0.5, "memory", writer),
                daemon=True,
            )
            cpu_tracker = threading.Thread(
                target=monitor_cpu,
                args=(stop_event, 0.5, "cpu", writer),
                daemon=True,
            )
            disk_tracker = threading.Thread(
                target=monitor_disk,
                args=(stop_event, 0.5, "disk", writer, path),
                daemon=True,
            )
            gpu_monitors = []
            for i, g in enumerate([0]):
                gpu_monitor = threading.Thread(
                    target=monitor_gpu,
                    args=(stop_event, 0.1, "gpu", writer, i),
                    daemon=True,
                )
                gpu_monitor.start()
                gpu_monitors.append(gpu_monitor)
            memory_tracker.start()
            cpu_tracker.start()
            disk_tracker.start()

        start = time.time()

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start

        if live_tracking:
            stop_event.set()
            memory_tracker.join()
            cpu_tracker.join()
            disk_tracker.join()
            for g in gpu_monitors:
                g.join()

        final_disk_io = psutil.disk_io_counters()
        final_disk_p = p.io_counters()
        bytes_read = final_disk_io.read_bytes - initial_disk_io.read_bytes
        chars_read = final_disk_p.read_chars - initial_disk_p.read_chars
        kilobytes_read = int(bytes_read / 1024)
        writer.add_scalar("kilobytes_read", kilobytes_read, slide_num)
        writer.add_scalar("chars_read", chars_read, slide_num)
        writer.add_scalar(
            "kilobytes_read_per_s", kilobytes_read / elapsed_time, slide_num
        )
        writer.add_scalar("chars_read_per_s", chars_read / elapsed_time, slide_num)
        writer.add_scalar("time_elapsed", elapsed_time, slide_num)

        return result

    return wrapper
