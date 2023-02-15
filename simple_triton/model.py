from google.protobuf.json_format import MessageToDict
import time
from tritonclient.utils import InferenceServerException
import json

def instance_group(count, kind="KIND_GPU", gpus=None):
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
        One of "KIND_GPU" for gpu serving, or "KIND_CPU" for cpu serving.
        Default value of "KIND_GPU" specifies that `count` models be hosted
        on each available gpu.
    gpus : list of int
        If specified, `count` instances will be hosted on each of the listed
        gpus. For example, [0, 1] would specify serving on gpus zero and one.
        Default value of `None` means that `count` instances will be served on
        each available gpu.
    """
    # gpus must be None if KIND_GPU
    if kind == "KIND_CPU":
        gpus = None
    if gpus is None:
        instance = {"count": count, "kind": kind}
    else:
        instance = {"count": count, "kind": kind, "gpus": gpus}

    return instance



def get_config_by_name(model_config, name):
        """Get input properties corresponding to the input
        with given `name`
        Parameters
        ----------
        model_config : dict
            dictionary object containing the model configuration
        name : str
            name of the input object
        Returns
        -------
        dict
            A dictionary containing all the properties for a given input
            name, or None if no input with this name exists
        """
        if 'input' in model_config:
            inputs = model_config['input']
            for input_properties in inputs:
                if input_properties['name'] == name:
                    return input_properties
                
        if 'output' in model_config:
            outputs = model_config['output']
            for output_properties in outputs:
                if output_properties['name'] == name:
                    return output_properties
                
        if 'max_batch_size' in model_config:
                max_batch_size = model_config['max_batch_size']
                if name == 'max_batch_size':
                    return max_batch_size

                
        if 'instance_group' in model_config:
            instance_group = model_config['instance_group']
            for output_properties in instance_group:
                if  name == 'instance_group':
                    return output_properties
                
        if 'execution_accelerators' in model_config:
            return model_config
            optimization = model_config['execution_accelerators']
            for output_properties in optimization:
                if  name == 'execution_accelerators':
                    return output_properties

        return None


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

def model_update(batch_config, config, optimization, client, model_name):
    """loaded model with updated configurations.

        The model configuration defines serving parameters like input, output, maximum batch size,
        dynamic batching, maximum queue delay, and maps instances to system cpu and
        gpu resources.

        Parameters
        ----------
        update : string
            updated model configuration parameters
        instance : string
            updated instance configuration parameters
        optimization : string 
            updated optimization configuration parameters
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
    config_model = model_config(client, model_name)

    max_batch_size = get_config_by_name(json.loads(batch_config), "max_batch_size")
    instance = instance_group(list(config.values())[0],
                              list(config.values())[1],
                              list(config.values())[2])
    optimization = get_config_by_name(json.loads(optimization), "execution_accelerators") # TBD: pass optimization as dict

    config_model["maxBatchSize"] = max_batch_size 
    config_model["instanceGroup"][0] = instance 
    config_model["optimization"] = optimization 
    
    client.load_model(model_name,config=json.dumps(config_model))
    return config_model


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
                    client.load_model(model_name, config=config)
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
