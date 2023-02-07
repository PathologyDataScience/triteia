from google.protobuf.json_format import MessageToDict
import time
import multiprocessing
import multiprocessing.queues
import numpy as np
from tritonclient.utils import InferenceServerException
from functools import partial


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
        output = self.dtype(np.random.uniform(size=(self.B, self.D)))
        return output


class TimedQueue(multiprocessing.queues.Queue):  # pragma: no cover
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


def instance_group(count, kind="KIND_GPU", gpus=None):  # pragma: no cover
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


def model_idle(client, model_name, idle=1000.0):
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
        The time window (milliseconds) after the last inference when a model
        is considered idle. The default value is a model is idle after 1000
        milisecodns have elapsed since the last inference.

    Returns
    -------
    status : bool
        Returns True is model is idle.
    """
    # use the client to get model statistics
    model_stats = client.get_inference_statistics(model_name)
    model_stats = MessageToDict(model_stats)
    # convert from protobuffer to dict
    try:
        print(
              model_stats["modelStats"][0]["lastInference"]
        )
        # calculate time elapsed since last inference (milliseconds)
        delta = round(time.time() * 1000) - int(
            model_stats["modelStats"][0]["lastInference"]
        )
        print(delta)
        # return idle status
        return delta > idle
    except:
        print("false")
        return True



class ModelTestsCases(object):
    def model_tests(
        self,
        client,
        client_close,
        model_name,
        batch_size,
        dimension_input,
        N,
        config=None,
        retries=5,
        wait=10e-3,
        block=True,
        timeout=1.0,
        verbose=True,
        model_dicts={},
        idle_check=False,
    ):
        """Load a model or update a model configuration on the Triton server.

        If the model is not loaded on the server, this model will attempt to load
        the model from the repository. If the config is provided, the model will be
        loaded with the provided config. If the config is not provided, the config
        will be auto generated by Triton. Calling this function for a loaded model
        will re-load the model with the provided config. In this case the model
        will be checked to see if it is idle before re-loading. This function will
        raise exceptions i  f there is an invalid configuration, or if problems are
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
            idle definition. Default value is True.
        timeout : float
            The timeout limit for waiting for model idle status. Default value is
            1 second.
        verbose : bool
            If True the function will emit status messages to stdout. Default value
            is False.
        """

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

        def _client_inputs(inputs, model_dict):
            """Generates tritonclient.grpc.InferInput objects for client.

            Given inputs and a model configuration and metadata, this function
            validates the inputs against the model expectations, and generates the
            InferInput objects used by the client to transmit inputs to triton.

            Parameters
            ----------
            inputs : list of numpy.ndarray
                A list of numpy arrays to input for model inference.
            model_dict : dict
                A dictionary describing the name, shape, and type of inputs and
                outputs, as well as maximum batch size.

            Outputs
            -------
            infer_inputs : list of tritonclient.grpc.InferInput
                A list of InferInput objects populated with data.
            """

            import tritonclient.grpc as grpcclient

            # # validate inputs against model expectations
            # self._validate_inputs(inputs, model_dict)

            # create InferInput objects for each model input
            infer_inputs = []
            for (provided, expected) in zip(inputs, model_dict["inputs"]):

                # create InferInput object
                iio = grpcclient.InferInput(
                    expected["name"], provided.shape, expected["datatype"]
                )

                # add numpy data to input
                # if self._np_to_api_types(provided.dtype) != expected["datatype"]:
                #     provided = provided.astype(self._api_to_np_types(expected["datatype"]))
                iio.set_data_from_numpy(provided)

                # add to infer_input list
                infer_inputs.append(iio)

            return infer_inputs

        def _client_outputs(model_dict):
            """Generates tritonclient.grpc.InferRequestedOutput objects for client.

            Parameters
            ----------
            model_dict : dict
                A dictionary describing the name, shape, and type of inputs and
                outputs, as well as maximum batch size.

            Outputs
            -------
            infer_outputs : list of tritonclient.grpc.InferRequestedOutput
                A list of InferRequestedOutput objects to capture inference
                results.
            """

            import tritonclient.grpc as grpcclient

            # create InferInput objects for each model input
            infer_outputs = [
                grpcclient.InferRequestedOutput(o["name"])
                for o in model_dict["outputs"]
            ]

            return infer_outputs

        # execute all client calls in a try block to catch exceptions
        def _callback(capture, result, error):
            """Callback for async_infer to capture result or error of inference
            request.

            Parameters
            ----------
            capture : list
                An empty list of
            result : grpcclient.InferResult
                The result of inference if successful.
            error : tritonclientutils.InferenceServerException
                An exception if inference failed. Otherwise None.
            """

            if error:
                capture.append((error, time.time()))
            else:
                capture.append((result, time.time()))

        def f(model_name, batch_size, dimension_input, N):

            sample = []

            if model_name not in model_dicts.keys():
                model_dict = {
                    **model_metadata(client, model_name),
                    "max_batch_size": model_config(client, model_name)["maxBatchSize"],
                }
                model_dicts[model_name] = model_dict
            else:
                model_dict = model_dicts[model_name]

            producer = iter(SimulatedProducer(batch_size, dimension_input, np.float16))

            print("Enqueuing inference jobs")
            for _ in range(N):
                data = next(producer)
                metadata = {"key": "random stuff"}
                inputs = _client_inputs(data, model_dict)

                # create outputs
                outputs = _client_outputs(model_dict)
                client.async_infer(
                    model_name=model_name,
                    inputs=inputs,
                    callback=partial(_callback, sample),
                    outputs=outputs,
                    client_timeout=timeout,
                )
                print("infer")

        def server_check(client):
            # initialize attempts
            attempts = 0
            while True:
                try:
                    response = client_close.is_server_ready()
                    if not response:
                        attempts += 1
                        if attempts > retries:
                            raise Exception(
                                "Triton server is not ready. Retry limit reached."
                            )
                        else:
                            time.sleep(wait)
                            continue
                    else:
                        continue
                except:
                    attempts += 1
                    if attempts > retries:
                        return
                        raise Exception(
                            "Triton server is not ready. Retry limit reached."
                        )
                    else:
                        time.sleep(wait)
                        continue

        def model_check(config=None):
            # initialize attempts
            attempts = 0
            while True:
                try:
                    # 2
                    if client.is_model_ready(model_name) and config is None:
                        if verbose:
                            print(f"load_model(): {model_name} is already loaded.")
                        return
                    # 3
                    if not client.is_model_ready(model_name):
                        if config is None:
                            if verbose:
                                print(
                                    f"load_model(): {model_name} is not loaded. Loading with auto-config."
                                )
                            client.load_model(model_name)
                            return
                        # 4
                        else:
                            if verbose:
                                print(
                                    f"load_model(): {model_name} is not loaded. Loading with provided config."
                                )
                            client.load_model(model_name, config)
                            return
                    # 5 : check if model is idle and can be unloaded
                    elif client.is_model_ready(model_name) and config is not None:
                        if verbose:
                            print(
                                f"load_model(): {model_name} is loaded. Re-loading with provided config."
                            )
                        if model_idle(client, model_name):
                            # client.unload_model(model_name)
                            # client.load_model(model_name, config)
                            print("Model is idle")
                            return
                        # 6 : if not idle, either increment attempts or check timeout
                        if not block:
                            attempts += 1
                            if attempts > retries:
                                raise Exception(
                                    f"Model {model_name} not idle. Retry limit reached."
                                )
                            time.sleep(wait)
                            continue
                        else:
                            if attempts == 0:
                                start = time.time()
                                attempts += 1
                            if time.time() - start > timeout:
                                raise Exception(
                                    f"Model {model_name} not idle. Block timeout elapsed."
                                )
                except InferenceServerException as e:
                    raise

        server_check(client_close)  # Server is not ready

        server_check(
            client
        )  # 1: server should be ready before models can be manipulated

        model_check(
            config=None
        )  #  2: model is loaded and configuration not provided - do nothing

        client.unload_model(model_name)  # to make sure load model check is executed

        model_check(config=None)  # 3: model is not loaded - attempt to load

        client.unload_model(model_name)  # to make sure load model check is executed

        model_check(
            config
        )  
        # 4: model is not loaded - attempt to load with provided config
        # start inference to check model idle condition

        f(model_name, batch_size, dimension_input, 1)
        model_check(config)  

        # 5. model is loaded and idle - reload model and with provided config
        # spawn inference process make model not idle
        p = multiprocessing.Process(
            target=f, args=(model_name, batch_size, dimension_input, N)
        )
        print("Start model inference")
        print("Start model check")
        for n in range(10):
            model_check(config)

