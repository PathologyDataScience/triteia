from google.protobuf.json_format import MessageToDict
import time
from tritonclient.utils import InferenceServerException
import json

def instance_group(model_name, count, kind="gpu", gpus=None):
    """Generates an instance count dictionary for use in a model config.

    A Triton configuration allows specification of resources used to serve a
    model. The instance group specifies the number of concurrent instances of
    a model to serve for a given set of resources. Resources can specify cpu
    or gpu hosting, or specific gpus. See Triton documentation for more
    details.

    Parameters
    ----------
    count : int
        The number of model instances to run concurrently.
    kind : str
        One of {"cpu", "gpu"}. Default value of "gpu" specifies that `count`
        models be hosted on each available gpu.
    gpus : list of int
        If specified, `count` instances will be hosted on each of the listed
        gpus. For example, [0, 1] would specify serving on gpus zero and one.
        Default value of `None` means that `count` instances will be served on
        each available gpu.
    """

    if not isinstance(count, int):
        # raise ValueError("argument 'count' must be int.")
        print()
    if kind.lower() not in ["cpu", "gpu"]:
        raise ValueError("argument 'kind' must be one of 'cpu', 'gpu'.")
    elif kind == "cpu":
        kind = "KIND_CPU"
    else:
        kind = "KIND_GPU"
    if gpus is not None:
        if not isinstance(gpus, list):
            raise ValueError("argument 'gpus' must be list of int.")
        for index in gpus:
            if not isinstance(index, int):
                raise ValueError("elements of 'gpus' must be int.")

    # set model name, count, and kind
    instance = {"name": model_name, "count": count, "kind": kind}
    if gpus is not None:
        instance["gpus"] = gpus

    return instance


def model_config(client, model_name):
    """Queries model config to retrieve triton model configuration.

    The model configuration defines serving parameters like maximum batch size,
    dynamic batching, maximum queue delay, and maps instances to system cpu and
    gpu resources.

    Parameters
    ----------
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server.
    model_name : string
        The name of the model to query as registered in triton.

    Returns
    -------
    config : dict
        A dictionary describing the model configuration. See Triton
        documentation for more details.
    """

    # get model config
    config = MessageToDict(client.get_model_config(model_name))

    return config["config"]


def model_idle(client, model_name, idle=1.0):
    """Determines if a model is idle based on time of last inference.

    Since we cannot query the model queue size of pending requests, we
    determine if a model is idle based on the time elapsed since the last
    completed inference.

    Parameters
    ----------
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server.
    model_name : string
        The name of the model to query as registered in triton.
    idle : float
        The time window (seconds) after the last inference when a model
        is considered idle. The default value is a model is idle after 1.
        second has elapsed since the last inference.

    Returns
    -------
    status : bool
        Returns True is model is idle.
    """

    # use the client to get model statistics
    model_stats = client.get_inference_statistics(model_name)

    # convert from protobuffer to dict
    model_stats = MessageToDict(model_stats)

    # get last inference time as float in seconds
    if "lastInference" in model_stats["modelStats"][0].keys():
        last_inference = float(model_stats["modelStats"][0]["lastInference"]) / 1000

        # calculate time elapsed since last inference seconds
        delta = time.time() - last_inference

        # return idle status
        return delta > idle

    else:  # model has zero inferences
        return True


def model_metadata(client, model_name):
    """Queries model metadata to retrieve model input/output signature.

    Parameters
    ----------
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server.
    model_name : string
        The name of the model to query as registered in triton.

    Returns
    -------
    metadata : dict
        A dictionary describing the name, shape, and type of inputs and
        outputs, as well as maximum batch size.
    """

    # get model metadata
    metadata = MessageToDict(client.get_model_metadata(model_name))

    return metadata


def model_update(
    client, model_name, max_batch_size=None, instances=None, trt=None, amp=False
):
    """Generate a valid configuration for model loading.

    The model configuration defines serving parameters like input/output names,
    types, and dimensions as well as model optimizations and other serving
    parameters like maximum batch size and system resources.

    Parameters
    ----------
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server.
    model_name : string
        The name of the model to query as registered in triton.
    max_batch_size : int
        The maximum batch size for the model. If None, do not add maxBatchSize
        to the generated config. Default value is None.
    instance : string
        updated instance configuration parameters
    trt : string
        updated optimization configuration parameters.
        Possible values are None, "FP32", "FP16"
    amp : bool
        In Automatic FP16 Optimization, TensorFlow has an option to provide
        FP16 optimization that can be enabled in the model configuration.

    Returns
    -------
    config : dict
        A dictionary describing the model configuration. See Triton
        documentation for more details.
    """

    def gpu_accelerator_status(config, name):
        # returns True if a gpuExecutionAccelerator with "name" value `name` is present

        if "optimization" in config:
            if "executionAccelerators" in config["optimization"]:
                if (
                    "gpuExecutionAccelerator"
                    in config["optimization"]["executionAccelerators"]
                ):
                    for d in config["optimization"]["executionAccelerators"][
                        "gpuExecutionAccelerator"
                    ]:
                        if "name" in d:
                            if d["name"] == name:
                                return True
                    return False
                else:
                    return False
            else:
                return False
        else:
            return False

    def gpu_accelerator_delete(config, name):
        # delete a gpuExecutionAccelerator and cleanup empty parents

        if gpu_accelerator_status(config, name):
            gpuexecacc = config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ]
            keep = []
            for i, d in enumerate(gpuexecacc):
                if "name" in enumerate(d):
                    if d["name"] != name:
                        keep.append(i)
            config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ] = [gpuexecacc[i] for i in keep]
            if (
                len(
                    config["optimization"]["executionAccelerators"][
                        "gpuExecutionAccelerator"
                    ]
                )
                == 0
            ):
                del config["optimization"]["executionAccelerators"][
                    "gpuExecutionAccelerator"
                ]
            if config["optimization"]["executionAccelerators"] == {}:
                del config["optimization"]["executionAccelerators"]
            if config["optimization"] == {}:
                del config["optimization"]

    def gpu_accelerator_trt(precision="FP16"):
        # returns a trt configuration
        if precision.upper() == "FP16" or precision.upper() == "FP32":
            return {
                "name": "tensorrt",
                "parameters": {"precision_mode": f"{precision.upper()}"},
            }
        else:
            raise ValueError(
                "precision must be one of None, numpy.float16, numpy.float32"
            )

    def gpu_accelerator_amp():
        # returns an amp configuration
        return {"name": "auto_mixed_precision"}

    def gpu_accelerator_add(config, accelerator=gpu_accelerator_trt("FP16")):
        # adds a gpu accelerator to a config
        if gpu_accelerator_status(config, accelerator["name"]):
            gpu_accelerator_delete(config, accelerator["name"])
        if "optimization" not in config:
            config["optimization"] = {}
        if "executionAccelerators" not in config["optimization"]:
            config["optimization"]["executionAccelerators"] = {}
        if (
            "gpuExecutionAccelerator"
            not in config["optimization"]["executionAccelerators"]
        ):
            config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ] = []
        config["optimization"]["executionAccelerators"][
            "gpuExecutionAccelerator"
        ].append(accelerator)

    # acquire the current model config
    config = model_config(client, model_name)

    # handle max batch size - verify that batch dimension exists
    batch_dim = [input["dims"][0] == "-1" for input in config["input"]]
    if all(batch_dim):
        config["maxBatchSize"] = str(max_batch_size)
    # else:
    #     raise Warning(
    #         f"Model {model_name} is not configured for batching, cannot set maxBatchSize"
    #     )

    # modify instance group if instances providedis not None
    if instances is not None:
        if isinstance(instances, dict):
            instances = [instances]
        elif isinstance(instances, list):
            if len(instances) == 0:
                del config["instanceGroup"]
        else:
            raise ValueError(
                "Instances must be an instance_group dict or list of dicts"
            )
    if not isinstance(amp, bool): 
      raise ValueError("trt must be a bool value")
    # remove amp if amp==False and amp is in current config
    if not amp:
        if gpu_accelerator_status(config, "auto_mixed_precision"):
            gpu_accelerator_delete(config, "auto_mixed_precision")

    # remove trt if trt is None and trt is in current config
    if trt is None:
        if gpu_accelerator_status(config, "tensorrt"):
            gpu_accelerator_delete(config, "tensorrt")

    # if amp is True, add only if trt is None
    if amp and trt is None:
        gpu_accelerator_add(config, gpu_accelerator_amp())

    # if trt is not none add trt, remove amp if necessary
    if trt is not None:
        if gpu_accelerator_status(config, "auto_mixed_precision"):
            gpu_accelerator_delete(config, "auto_mixed_precision")
        gpu_accelerator_add(config, gpu_accelerator_trt(trt))

    # apply model update
    load_model(client, model_name, config=json.dumps(config), block=True)


def load_model(
    client,
    model_name,
    config=None,
    retries=5,
    wait=10e-3,
    block=False,
    timeout=1.0,
    verbose=False,
):
    """Load a model or update a model configuration on the Triton server.

    If the model is not loaded on the server, this model will attempt to load
    the model from the repository. If the config is provided, the model will be
    loaded with the provided config. If the config is not provided, the config
    will be auto generated by Triton. Calling this function for a loaded model
    will re-load the model with the provided config. In this case the model
    will be checked to see if it is idle before re-loading. This function will
    raise exceptions if there is an invalid configuration, or if problems are
    encountered loading or unloading the model.

    Parameters
    ----------
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server.
    model_name : string
        The name of the model to query as registered in triton.
    config : dict
        A dictionary defining the model configuration defining the input/output
        names and type and model resources. Default value of None will result
        in Triton generating a configuration. See Triton documentation for more
        details.
    retries : int
        The maximum number of attempts for each request. Default value is 5.
    wait : float
        The time to wait between failed attempts. Default value is 0.1 seconds.
    block : bool
        If True, block until the model is idle. See check_stats() for model
        idle definition. Loading can either retry or block, but not both.
        Default value is False for no blocking (will use retry instead).
    timeout : float
        The timeout limit for waiting for model idle status. Default value is
        1 second.
    verbose : bool
        If True the function will emit status messages to stdout. Default value
        is False.
    """

    # initialize attempts
    attempts = 0

    while True:
        # execute all client calls in a try block to catch exceptions
        try:
            # server should be live before models can be manipulated
            if not client.is_server_live():
                attempts += 1
                if attempts > retries:
                    raise InferenceServerException(
                        "Triton server is not ready. Retry limit reached."
                    )
                else:
                    time.sleep(wait)
                    continue

            # model is loaded and configuration not provided - do nothing
            if client.is_model_ready(model_name) and config is None:
                if verbose:
                    print(f"load_model(): {model_name} is already loaded.")
                return

            # model is not loaded - attempt to load
            if not client.is_model_ready(model_name):
                if config is None:
                    if verbose:
                        print(
                            f"load_model(): {model_name} is not loaded. Loading with auto-config."
                        )
                    client.load_model(model_name)
                    return
                else:
                    if verbose:
                        print(
                            f"load_model(): {model_name} is not loaded. Loading with provided config."
                        )
                    ç
                    return

            # reload model and with provided config
            elif client.is_model_ready(model_name) and config is not None:
                if verbose:
                    print(
                        f"load_model(): {model_name} is loaded. Re-loading with provided config."
                    )

                # check if model is idle and can be unloaded
                if model_idle(client, model_name):
                    client.unload_model(model_name)
                    client.load_model(model_name, config=config)
                    return

                # if not idle, either increment attempts or check timeout
                if not block:
                    attempts += 1
                    if attempts > retries:
                        raise InferenceServerException(
                            f"Model {model_name} not idle. Retry limit reached."
                        )
                    time.sleep(wait)
                    continue
                else:
                    if attempts == 0:
                        start = time.time()
                        attempts += 1
                    if time.time() - start > timeout:
                        raise InferenceServerException(
                            f"Model {model_name} not idle. Block timeout elapsed."
                        )

        except InferenceServerException as e:
            raise
