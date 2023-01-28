from functools import partial
import multiprocessing
import multiprocessing.queues
from multiprocessing import Process
import numpy as np
import os
import sys
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
        """

        if api_type == "FP32":
            return np.float32
        elif api_type == "FP16":
            return np.float16
        elif api_type == "FLOAT64":
            return np.float64
        elif api_type == "UINT8":
            return np.uint8
        elif api_type == "UINT16":
            return np.uint16
        elif api_type == "UINT32":
            return np.uint32
        elif api_type == "UINT64":
            return np.uint64
        elif api_type == "INT8":
            return np.int8
        elif api_type == "INT16":
            return np.int16
        elif api_type == "INT32":
            return np.int32
        elif api_type == "INT64":
            return np.int64
        elif api_type == "BOOL":
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
            return "FP32"
        elif dtype == np.float16:
            return "FP16"
        elif dtype == np.float64:
            return "FLOAT64"
        elif dtype == np.uint8:
            return "UINT8"
        elif dtype == np.uint16:
            return "UINT16"
        elif dtype == np.uint32:
            return "UINT32"
        elif dtype == np.uint64:
            return "UINT64"
        elif dtype == np.int8:
            return "INT8"
        elif dtype == np.int16:
            return "INT16"
        elif dtype == np.int32:
            return "INT32"
        elif dtype == np.int64:
            return "INT64"
        elif dtype == np.bool:
            return "BOOL"
        else:
            raise ValueError(f"Unrecognized type '{str(dtype)}'")

    def _model_metadata_config(self, model_name):
        """Queries triton to get model input and output names, shapes, types,
        and maximum batch size.

        Parameters
        ----------
        model_name : string
            The name of the model to query as registered in triton.

        Returns
        -------
        model_dict : dict
            A dictionary describing the name, shape, and type of inputs and
            outputs, as well as maximum batch size.
        """

        # get model metadata
        try:
            metadata = self.client.get_model_metadata(model_name)
        except InferenceServerException as e:
            print("Failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        # get model config
        try:
            config = self.client.get_model_config(model_name)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        # capture input names, shapes, and types
        model_inputs = []
        for i in metadata.inputs:
            if i.shape[0] == -1:
                shape = [None, *i.shape[1:]]
            else:
                shape = i.shape
            model_inputs.append({"name": i.name, "type": i.datatype, "shape": shape})

        # capture output names, shapes, and types
        model_outputs = []
        for o in metadata.outputs:
            if o.shape[0] == -1:
                shape = [None, *o.shape[1:]]
            else:
                shape = o.shape
            model_outputs.append({"name": o.name, "type": o.datatype, "shape": shape})

        # get max batch size
        max_batch_size = config.config.max_batch_size

        # capture outputs in dictionary
        model_dict = {
            "inputs": model_inputs,
            "outputs": model_outputs,
            "max_batch_size": max_batch_size,
            "name": model_name,
        }

        return model_dict

    def print_pending(self):
        """Prints current state of self.pending for debugging."""

        # print the process ID
        print(f"Pending inferences for {os.getpid()}", flush=True)

        # for each line of the queue, print the request ID and elapsed time
        for i, request in enumerate(self.pending):
            print(f"\t{i}\t{time.time()-request['elapsed_retrieval']}", flush=True)

    def _validate_inputs(self, inputs, model_dict):
        """Validate inputs against model config and metadata.

        Parameters
        ----------
        inputs : list of numpy.ndarray
            A list of numpy arrays to input for model inference.

        model_dict : dict
            A dictionary describing the name, shape, and type of inputs and
            outputs, as well as maximum batch size.

        See also
        --------
        _model_metadata_config, _model_inputs
        """

        # check number of inputs
        if len(inputs) != len(model_dict["inputs"]):
            raise Exception(
                f"Model {model_dict['name']} expects {len(model_dict['inputs'])} inputs, received {len(inputs)}."
            )

        # check input shapes
        for i, (provided, expected) in enumerate(zip(inputs, model_dict["inputs"])):
            if expected["shape"][0] is None:
                if not np.array_equal(
                    np.array(np.shape(provided)[1:], dtype=np.int32),
                    np.array(expected["shape"][1:], dtype=np.int32),
                ):
                    raise Exception(
                        f"Model {model_dict['name']} {model_dict['inputs'][i]['name']} has shape {expected}."
                    )
                # if provided.shape[0] > model_dict["max_batch_size"]:
                #     raise Exception(
                #         f"Model {model_dict['name']} has max batch size {model_dict['max_batch_size']}."
                #     )

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
            A dictionary describing the name, shape, and type of inputs and
            outputs, as well as maximum batch size.

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
        for (provided, expected) in zip(inputs, model_dict["inputs"]):

            # create InferInput object
            iio = grpcclient.InferInput(
                expected["name"], provided.shape, expected["type"]
            )

            # add numpy data to input
            if self._np_to_api_types(provided.dtype) != expected["type"]:
                provided = provided.astype(self._api_to_np_types(expected["type"]))
            iio.set_data_from_numpy(provided)

            # add to infer_input list
            infer_inputs.append(iio)

        return infer_inputs

    def _client_outputs(self, model_dict):
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
            grpcclient.InferRequestedOutput(o["name"]) for o in model_dict["outputs"]
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
                            self.model_dicts[request["model_name"]]["outputs"]
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
            model_dict = self._model_metadata_config(model_name)
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
        qin,
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
        qin : multiprocessing.Queue
            Input queue containing samples for inference. Each sample is a
            2-tuple or 3-tuple containing the model name (str), a list of numpy
            arrays for the model inputs (list of np.ndarray), and an optional
            dictionary of sample metadata that will stay linked to the
            inference request and result.
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
        self.qin = qin
        self.qout = qout
        self.limit = limit
        self.timeout = timeout
        self.rest = rest
        self.pre = pre
        self.post = post

        # initialize list to hold performance data
        self.time_inference = []

    def run(self):

        # set flag indicating qin stop signal received
        stop = False

        # create requests object
        req = Requests(self.url, self.limit)

        # loop until exit signal received from calling process
        while True:

            # fill input queue with requests up to limit
            if not stop:
                for i in range(self.limit - len(req.pending)):

                    # pull sample
                    sample, t_put, t_get = self.qin.get()

                    # check if stop signal, otherwise pack dictionary
                    if sample is None:
                        stop = True
                        break
                    else:
                        request = {
                            "model_name": sample[0],
                            "inputs": sample[1],
                            "times": {"qin_put": t_put, "qin_get": t_get},
                        }

                    # add metadata if present
                    if len(request) == 3:
                        request["metadata"] = sample[2]

                    # apply preprocessing function
                    # TBD

                    # fill requests if stop signal not received
                    req.insert(request, self.timeout)

            # check pending requests
            completed = req.check(block=False)

            # put completed post-processed requests into queue
            for inference in completed:

                # # apply postprocessing function
                # if len(inference["result"]):
                # TBD

                # place in queue
                self.qout.put(inference)

            # check if done
            if len(req.pending) == 0 and stop:
                break

            # sleep
            time.sleep(self.rest)

        return
