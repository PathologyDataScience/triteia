from functools import partial
from simple_triton.model import model_config, model_metadata
import multiprocessing
import multiprocessing.queues
from multiprocessing import Process
import numpy as np
import os
import time
from tritonclient.utils import InferenceServerException


class Requests(object):
    """A class to manage inference server requests.

    The class maintains a list of pending requests, ordered by submission time.
    It can be used to check on the status of pending requests and to insert
    new requests.
    """

    def __init__(self, url, limit, retries=5, verbose=False):
        """Construct

        Parameters
        ----------
        url : string
            A url for the triton GRPC port used to submit requests.
        limit : int
            The maximum number of concurrent pending requests to allow.
        retries : int
            The maximum number of attempts for each request.
        verbose : bool
            True updates console with inference progress and exceptions.
            Default value is False.
        """

        import tritonclient.grpc as grpcclient

        # create GRPC client
        try:
            self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
        except Exception as e:
            print("context creation failed: " + str(e), flush=True)

        self.limit = limit
        self.retries = retries
        self.verbose = verbose
        self.model_dicts = {}
        self.pending = []

    def _api_to_np_types(self, api_type):
        """Converts triton API type string to numpy dtype.

        Parameters
        ----------
        api_type : str
            The type string for the triton client API.

        Returns
        -------
        dtype : numpy.dtype
            The corresponding numpy dtype.

        Notes
        -----
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
        """

        if api_type == "TYPE_FP32":
            return np.float32
        elif api_type == "TYPE_FP16":
            return np.float16
        elif api_type == "TYPE_FLOAT64":
            return np.float64
        elif api_type == "TYPE_UINT8":
            return np.uint8
        elif api_type == "TYPE_UINT16":
            return np.uint16
        elif api_type == "TYPE_UINT32":
            return np.uint32
        elif api_type == "TYPE_UINT64":
            return np.uint64
        elif api_type == "TYPE_INT8":
            return np.int8
        elif api_type == "TYPE_INT16":
            return np.int16
        elif api_type == "TYPE_INT32":
            return np.int32
        elif api_type == "TYPE_INT64":
            return np.int64
        elif api_type == "TYPE_BOOL":
            return np.bool
        else:
            raise ValueError(f"Unrecognized type '{str(api_type)}'")

    def _np_to_api_types(self, dtype):
        """Converts numpy dtype to triton API type string.

        Parameters
        ----------
        dtype : numpy.dtype
            A numpy dtype.

        Returns
        -------
        api_type : str
            The corresponding type string for the triton client API.
        """

        if dtype == np.float32:
            return "TYPE_FP32"
        elif dtype == np.float16:
            return "TYPE_FP16"
        elif dtype == np.float64:
            return "TYPE_FLOAT64"
        elif dtype == np.uint8:
            return "TYPE_UINT8"
        elif dtype == np.uint16:
            return "TYPE_UINT16"
        elif dtype == np.uint32:
            return "TYPE_UINT32"
        elif dtype == np.uint64:
            return "TYPE_UINT64"
        elif dtype == np.int8:
            return "TYPE_INT8"
        elif dtype == np.int16:
            return "TYPE_INT16"
        elif dtype == np.int32:
            return "TYPE_INT32"
        elif dtype == np.int64:
            return "TYPE_INT64"
        elif dtype == np.bool:
            return "TYPE_BOOL"
        else:
            raise ValueError(f"Unrecognized type '{str(dtype)}'")

    def print_pending(self):
        """Prints current state of self.pending for debugging."""

        # print the process ID
        print(f"Pending inferences for {os.getpid()}", flush=True)

        # for each line of the queue, print the request ID and elapsed time
        for i, request in enumerate(self.pending):
            print(f"\t{i}\t{time.time()-request['elapsed_retrieval']}", flush=True)

    def _validate_inputs(self, inputs, model_dict, strict_types=False):
        """Validate inputs against model config and metadata.

        Parameters
        ----------
        inputs : list of numpy.ndarray
            A list of numpy arrays to input for model inference.
        model_dict : dict
            A model configuration dictionary describing the name, shape, and 
            type of inputs and outputs, and the maximum batch size if defined.
        strict_types : bool
            Enforce strict types on model inputs. _client_inputs will
            cast inputs to the correct type, so this is not necessary.
            Default value is False.

        See also
        --------
        model_metadata
        """

        # validate number of inputs
        if len(inputs) != len(model_dict["input"]):
            raise Exception(
                (
                    f"Model {model_dict['name']} expects {len(model_dict['input'])} "
                    f"inputs, received {len(inputs)}."
                )
            )

        # if model is batching -> batch dimension implied
        if "maxBatchSize" in model_dict:
            if model_dict["maxBatchSize"] > 0:
                batching = True
                batch_size = inputs[0].shape[0]
            else:
                batching = False
        else:
            batching = False

        def _compare_dims(provided, expected):
            return all(
                [True if e == -1 else p == e for (p, e) in zip(provided, expected)]
            )

        def _int(shape):
            return [int(s) for s in shape]

        # validate input types
        if strict_types:
            for i, (provided, expected) in enumerate(zip(inputs, model_dict["input"])):
                if self._np_to_api_types(provided.dtype) != expected["dataType"]:
                    raise Exception(
                        (
                            f"Model {model_dict['name']} input "
                            f"{model_dict['input'][i]['name']} "
                            f"expects type {expected['dataType']}, received input with "
                            f"type {provided.dtype}."
                        )
                    )

        # validate input shapes
        for i, (provided, expected) in enumerate(zip(inputs, model_dict["input"])):
            if batching:  # check uniform batch sizing and max_batch_size limit
                offset = 1
            else:
                offset = 0
            if len(provided.shape) != len(
                expected["dims"]
            ) + offset or not _compare_dims(
                provided.shape[offset:], _int(expected["dims"])
            ):
                if batching:
                    output = [-1, *_int(expected["dims"])]
                else:
                    output = _int(expected["dims"])
                raise Exception(
                    (
                        f"Model {model_dict['name']} input "
                        f"{model_dict['input'][i]['name']} "
                        f"expects shape {output}, received input with "
                        f"shape {list(provided.shape)}."
                    )
                )

        # validate batching
        if batching:
            for i, (provided, expected) in enumerate(zip(inputs, model_dict["input"])):
                if provided.shape[0] != batch_size:
                    raise Exception(
                        (
                            f"Non-uniform batch size of inputss. Expected {batch_size}, "
                            f"found {provided.shape[0]}."
                        )
                    )
                if provided.shape[0] > model_dict["maxBatchSize"]:
                    raise Exception(
                        (
                            f"Model {model_dict['name']} max batch size "
                            f"{model_dict['maxBatchSize']} exceeded."
                        )
                    )

    def _client_inputs(self, inputs, model_dict):
        """Generates tritonclient.grpc.InferInput objects for client.

        Given inputs and a model configuration and metadata, this function
        validates the inputs against the model expectations, and generates the
        InferInput objects used by the client to transmit inputs to triton.

        Parameters
        ----------
        inputs : list of numpy.ndarray
            A list of numpy arrays to input for model inference.
        model_dict : dict
            A model configuration dictionary describing the name, shape, and 
            type of inputs and outputs, and the maximum batch size if defined.

        Outputs
        -------
        infer_inputs : list of tritonclient.grpc.InferInput
            A list of InferInput objects populated with data.
        """

        import tritonclient.grpc as grpcclient

        # validate inputs against model expectations
        self._validate_inputs(inputs, model_dict)

        # create InferInput objects for each model input
        infer_inputs = []
        for provided, expected in zip(inputs, model_dict["input"]):
            # create InferInput object
            iio = grpcclient.InferInput(
                expected["name"], provided.shape, expected["dataType"].split("TYPE_")[1]
            )

            # add numpy data to input
            if self._np_to_api_types(provided.dtype) != expected["dataType"]:
                provided = provided.astype(self._api_to_np_types(expected["dataType"]))
            iio.set_data_from_numpy(provided)

            # add to infer_input list
            infer_inputs.append(iio)

        return infer_inputs

    def _client_outputs(self, model_dict):
        """Generates tritonclient.grpc.InferRequestedOutput objects for client.

        Parameters
        ----------
        model_dict : dict
            A model configuration dictionary describing the name, shape, and 
            type of inputs and outputs, and the maximum batch size if defined.

        Outputs
        -------
        infer_outputs : list of tritonclient.grpc.InferRequestedOutput
            A list of InferRequestedOutput objects to capture inference
            results.
        """

        import tritonclient.grpc as grpcclient

        # create InferRequestedOutput objects for each model output
        infer_outputs = [
            grpcclient.InferRequestedOutput(o["name"]) for o in model_dict["output"]
        ]

        return infer_outputs

    def _callback(self, capture, result, error):
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

    def check(self, block=True, wait=100e-3):
        """Check for completion of pending requests.

        Parameters
        ----------
        block : bool
            If True, the function will block until at least one request
            completes. Default value is True.
        wait : float
            If block is True, this is the interval to wait until checking
            requests again.

        Returns
        -------
        completed : list of tuple
            A list containing the results of completed inference requests.
        """

        # iterate through list of requests, checking who is finished
        def request_loop():
            # initialize lists of completed requests, requests to delete, requests to retry
            completed = []
            delete = []
            retry = []

            # check each pending result for completion
            for i, request in enumerate(self.pending):
                # request is complete if value
                if len(request["result"]):
                    # add request to list for deletion
                    delete.append(i)

                    # unpack request result into output, time
                    completion_time = request["result"][0][1]
                    results = request["result"][0][0]

                    # record inference and retrieval times
                    request["times"]["completed"] = completion_time
                    request["times"]["retrieved"] = time.time()

                    # inference generated an exception
                    if type(results) == InferenceServerException:
                        # clear result
                        request["result"] = []

                        # capture error in request
                        if request["attempts"] == 1:
                            request["errors"] = []
                        request["errors"].append(results.message())

                        # make another attempt if retry limit has not been reached
                        if request["attempts"] < self.retries:
                            # add request to list of retries to be processed
                            retry.append(request)

                        else:
                            # add request to output list
                            completed.append(request)

                    # inference generated a result
                    else:
                        # convert responses to numpy arrays
                        for j, output in enumerate(
                            self.model_dicts[request["model_name"]]["output"]
                        ):
                            request["result"][j] = results.as_numpy(output["name"])

                        # add request to output list
                        completed.append(request)

            return completed, delete, retry

        # if no blocking, iterate through list once and return
        if not block:
            completed, delete, retry = request_loop()
        else:
            while True:
                completed, delete, retry = request_loop()
                if len(delete):
                    break
                time.sleep(wait)

        # delete completed entries from list
        for i in sorted(delete, reverse=True):
            del self.pending[i]

        # retry errors
        for request in retry:
            self.insert(request)

        return completed

    def insert(self, sample, timeout=None):
        """Insert an inference request for submission to triton.

        Parameters
        ----------
        sample : dict

        timeout : float
            Timeout for the request in seconds. Default value is None.
        """

        # process input sample - if list or tuple
        if not isinstance(sample, dict):
            raise ValueError(
                "Input 'sample' is a dict with required keys 'model_name' and 'inputs'."
            )
        if "model_name" not in sample.keys():
            raise ValueError("Input 'sample' must have key 'model_name'.")
        if "inputs" not in sample.keys():
            raise ValueError("Input 'sample' must have key 'inputs'.")

        # get model name
        model_name = sample["model_name"]

        # check if model_dict has been previously generated for model_name
        if model_name not in self.model_dicts.keys():
            model_dict = model_config(self.client, model_name)
            self.model_dicts[model_name] = model_dict
        else:
            model_dict = self.model_dicts[model_name]

        # create InputData objects based on data shape
        inputs = self._client_inputs(sample["inputs"], model_dict)

        # create outputs
        outputs = self._client_outputs(model_dict)

        # initialize output
        sample["result"] = []

        # add submission time to request
        sample["times"]["submitted"] = time.time()

        # increment attempts
        if "attempts" not in sample.keys():
            sample["attempts"] = 0
        sample["attempts"] = sample["attempts"] + 1

        # submit request
        self.client.async_infer(
            model_name=model_name,
            inputs=inputs,
            callback=partial(self._callback, sample["result"]),
            outputs=outputs,
            client_timeout=timeout,
        )

        # append request to list
        self.pending.append(sample)


class InferenceRunner(Process):
    """InferenceRunner"""

    def __init__(
        self,
        url,
        model,
        dataset,
        qout,
        limit=10,
        rest=1e-2,
        timeout=None,
        pre=None,
        post=None,
        verbose=False,
    ):
        """InferenceRunner constructor.

        Parameters
        ----------
        url : string
            The inference server url.
        model : string
            The model name.
        dataset : object
            An iterator producing batched samples and metadata. Each batch
            is a 2-tuple containing a list of numpy arrays for the model
            inputs (list of np.ndarray), and an optional dictionary of
            sample metadata that will stay linked to the inference request
            and result.
        qout : multiprocessing.Queue
            Output queue receiving completed inference requests and process
            summary information on process completion.
        limit : int
            The maximum allowable pending requests. Default value is 10.
        rest : float
            The resting period for the InferenceRunner process. The process
            will rest for this period (seconds) after submitting and checking
            inference requests. Default value is 10 milliseconds.
        timeout : float
            The request timeout limit (seconds). This is an input argument
            to the triton GRPC client async_infer inferface.
        pre : function
            A preprocessing function to apply to samples prior to inference.
            Default value is None.
        post : function
            A postprocessing function to apply to inference results. Default
            value is None.
        verbose : bool
            True updates console with inference progress and exceptions.
            Default value is False.
        """

        multiprocessing.Process.__init__(self)

        # capture input arguments
        self.url = url
        self.model = model
        self.dataset = dataset
        self.qout = qout
        self.limit = limit
        self.timeout = timeout
        self.rest = rest
        if pre is None:
            self.pre = lambda x: x
        else:
            self.pre = pre
        if post is None:
            self.post = lambda x: x
        else:
            self.post = post

        # initialize list to hold performance data
        self.time_inference = []

    def run(self):
        # set flag indicating qin stop signal received
        stop = False

        # create requests object
        req = Requests(self.url, self.limit)

        # loop until iterator is exhausted
        while True:
            if not stop:
                # draw samples, preprocess, and submit for inference up to limit
                for i in range(self.limit - len(req.pending)):
                    try:
                        t_put = time.time()
                        sample, metadata = next(self.dataset)
                        t_get = time.time()
                    except StopIteration as e:
                        stop = True
                    if not stop:
                        sample = self.pre(sample)
                        request = {
                            "model_name": self.model,
                            "inputs": [sample],
                            "metadata": metadata,
                            "times": {"qin_put": t_put, "qin_get": t_get},
                        }
                        req.insert(request, self.timeout)
                    else:
                        break

            # check pending requests
            completed = req.check(block=False)

            # put completed post-processed requests into queue
            for inference in completed:
                # apply postprocessing function
                # if len(inference["result"]):
                # TBD

                # remove inputs
                del inference["inputs"]

                # place in queue
                self.qout.put(inference)

            # if done send stop signal
            if len(req.pending) == 0 and stop:
                self.qout.put(None)
                break

            # sleep
            time.sleep(self.rest)

        return
