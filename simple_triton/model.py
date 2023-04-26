from google.protobuf.json_format import MessageToDict
import json
from simple_triton.utils import create_client
import time
from tritonclient.utils import InferenceServerException


class TritonModel(object):
    """An object for controlling models on Triton server.

    The Model class performs loading/unloading models and updating
    hosted model configurations. The model class is also used to
    check model idle status, and to retrieve the model configuration
    and metadata.

    Parameters
    ----------
    model_name : string
        The name of the model to query as hosted in triton or stored in
        the model repository.
    url : string
        The url for the remote-procedure call port of the Triton server.
        Default value is "localhost:8001".

    Attributes
    ----------
    model_name : str
        The name of the model to query as registered in triton.

    Methods
    -------
    get_config(verbose=False)
        Retrieve the configuration for a loaded model.
    is_idle(idle=1.0)
        Check if model is idle based on time since last inference.
    is_loaded()
        Check if model is loaded on the server.
    get_metadata(verbose=False)
        Retrieve metadata for a loaded model.
    load(config=None, retries=5, wait=0.01, block=False, timeout=1.0, verbose=False)
        Load a model from the model repository.
    unload(retries=5, wait=0.01, block=False, timeout=1.0, verbose=False)
        Unload a loaded model.

    Notes
    -----
    To load models while Triton is running, the Triton server has to be launched
    with `--model-control-mode=explicit`.

    References
    ----------
    Model Control https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md
    """

    def __init__(self, model_name, url="localhost:8001"):
        self.model_name = model_name
        self.url = url

    def get_config(self, verbose=False):
        """Queries model config to retrieve triton model configuration.

        The model configuration defines serving parameters like maximum batch size,
        dynamic batching, maximum queue delay, and maps instances to system cpu and
        gpu resources.

        Parameters
        ----------
        verbose : bool
            Whether to display client activity in stdout. Default value is
            `False`.

        Returns
        -------
        config : dict
            A dictionary describing the model configuration. See Triton
            documentation for more details.
        """

        client = create_client(self.url, verbose)
        config = MessageToDict(client.get_model_config(self.model_name))
        return config["config"]

    def is_idle(self, idle=1.0):
        """Determines if a model is idle based on time of last inference.

        Since we cannot query the model queue size of pending requests, we
        determine if a model is idle based on the time elapsed since the last
        completed inference.

        Parameters
        ----------
        idle : float
            The time window (seconds) after the last inference when a model
            is considered idle. The default value is a model is idle after 1.
            second has elapsed since the last inference.

        Returns
        -------
        status : bool
            Returns True is model is idle.
        """

        client = create_client(self.url)
        model_stats = client.get_inference_statistics(self.model_name)
        model_stats = MessageToDict(model_stats)

        # get last inference time as float in seconds
        if "lastInference" in model_stats["modelStats"][0].keys():
            last_inference = float(model_stats["modelStats"][0]["lastInference"]) / 1000
            delta = time.time() - last_inference
            return delta > idle
        else:  # model has zero inferences
            return True

    def is_loaded(self):
        """Queries server to test if model is loaded and ready.

        Returns
        -------
        status : bool
            Returns True is model is loaded and ready.
        """

        client = create_client(self.url)
        return client.is_model_ready(self.model_name)

    def get_metadata(self, verbose=False):
        """Queries model metadata to retrieve model input/output signature.

        Parameters
        ----------
        verbose : bool
            Whether to display client activity in stdout. Default value is
            `False`.

        Returns
        -------
        metadata : dict
            A dictionary describing the name, shape, and type of inputs and
            outputs, as well as maximum batch size.
        """

        client = create_client(self.url, verbose)
        metadata = MessageToDict(client.get_model_metadata(self.model_name))
        return metadata

    def load(
        self, config=None, retries=5, wait=0.01, block=False, timeout=1.0, verbose=False
    ):
        """Load a model or update a model configuration on the Triton server.

        If not currently loaded, load the model from the repository. If a
        configuration is provided the model will be loaded/reloaded with this
        configuration. If the configuration is not provided, Triton will
        automatically generate a new configuration. When the model is loaded,
        the function will first check to see if it is idle before re-loading.
        Exceptions will be raised if there is an invalid configuration, or if
        other problems are encountered loading or unloading the model.

        Parameters
        ----------
        config : dict
            A dictionary defining the model configuration defining the input/output
            names and type and model resources. Default value `None` will result
            in Triton generating a configuration. See Triton documentation for
            details.
        retries : int
            The maximum number of attempts for each request. Default is 5.
        wait : float
            The time to wait between failed attempts. Default is 0.1 seconds.
        block : bool
            If True, block until the model is idle. See check_stats() for model
            idle definition. Loading can either retry or block, but not both.
            Default is False for no blocking (will use retry instead).
        timeout : float
            The timeout limit for waiting for model idle status. Default is
            1 second.
        verbose : bool
            If True the client will emit status messages to stdout. Default
            is False.
        """

        client = create_client(self.url, verbose)
        attempts = 0
        while True:
            try:  # execute all client calls in a try block to catch exceptions
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
                if client.is_model_ready(self.model_name) and config is None:
                    if verbose:
                        print(f"load_model(): {self.model_name} is already loaded.")
                    return

                # model is not loaded - attempt to load
                if not client.is_model_ready(self.model_name):
                    if config is None:
                        if verbose:
                            print(
                                f"load_model(): {self.model_name} is not loaded. Loading with auto-config."
                            )
                        client.load_model(self.model_name)
                        return
                    else:
                        if verbose:
                            print(
                                f"load_model(): {self.model_name} is not loaded. Loading with provided config."
                            )
                        client.load_model(self.model_name, config=json.dumps(config))
                        return

                # reload model and with provided config
                elif client.is_model_ready(self.model_name) and config is not None:
                    if verbose:
                        print(
                            f"load_model(): {self.model_name} is loaded. Re-loading with provided config."
                        )

                    # check if model is idle and can be unloaded
                    if self.is_idle():
                        client.unload_model(self.model_name)
                        client.load_model(self.model_name, config=json.dumps(config))
                        return

                    # if not idle, either increment attempts or check timeout
                    if not block:
                        attempts += 1
                        if attempts > retries:
                            raise InferenceServerException(
                                f"Model {self.model_name} not idle. Retry limit reached."
                            )
                        time.sleep(wait)
                        continue
                    else:
                        if attempts == 0:
                            start = time.time()
                            attempts += 1
                        if time.time() - start > timeout:
                            raise InferenceServerException(
                                f"Model {self.model_name} not idle. Block timeout elapsed."
                            )
            except InferenceServerException as e:
                raise

    def unload(self, retries=5, wait=0.01, block=False, timeout=1.0, verbose=False):
        """Unload a model from the Triton server.

        Parameters
        ----------
        verbose : bool
            If True the client will emit status messages to stdout. Default
            is False.
        """

        client = create_client(self.url, verbose)
        attempts = 0
        while True:
            try:  # execute all client calls in a try block to catch exceptions
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

                # model is loaded
                if client.is_model_ready(self.model_name):
                    # check if model is idle and can be unloaded
                    if self.is_idle():
                        client.unload_model(self.model_name)
                        return

                    # if not idle, either increment attempts or check timeout
                    if not block:
                        attempts += 1
                        if attempts > retries:
                            raise InferenceServerException(
                                f"Model {self.model_name} not idle. Retry limit reached."
                            )
                        time.sleep(wait)
                        continue
                    else:
                        if attempts == 0:
                            start = time.time()
                            attempts += 1
                        if time.time() - start > timeout:
                            raise InferenceServerException(
                                f"Model {self.model_name} not idle. Block timeout elapsed."
                            )

                # model is not loaded - do nothing
                else:
                    return
            except InferenceServerException as e:
                raise
