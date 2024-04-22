import argparse
from functools import partial
import numpy as np
import os
from simple_triton.model import TritonModel
from simple_triton.tile_iterators import SharedNumpyArray
from simple_triton.utils import create_client
from simple_triton.config import ConfigBuilder
from simple_triton.feature_extraction import inference, study
from simple_triton.tile_iterators import TiffPrefetch
from mil.io.writer import write_record
import tensorflow as tf

import time
import large_image_source_tiff
from tritonclient.utils import (
    InferenceServerException,
    triton_to_np_dtype,
    np_to_triton_dtype,
)


ARRAY_TYPES = (np.ndarray, SharedNumpyArray)


class Requests(object):
    """A class to manage inference server requests.

    The class maintains a list of pending requests, ordered by submission time.
    It can be used to check on the status of pending requests and to insert
    new requests.
    """

    def __init__(self, url="localhost:8001", limit=10, retries=5, verbose=False):
        """Construct

        Parameters
        ----------
        url : string
            The url for the remote-procedure call port of the Triton server.
            Default value is "localhost:8001".
        limit : int
            The maximum number of concurrent pending requests to allow. Default
            value is 10.
        retries : int
            The maximum number of attempts for each request. Default value is 5.
        verbose : bool
            True updates console with inference progress and exceptions.
            Default value is False.
        """

        self.limit = limit
        self.model_dicts = {}
        self.pending = []
        self.retries = retries
        self.client = create_client(url, verbose)
        self.url = url
        self.verbose = verbose

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
        All types listed in model configurations are prepended with "TYPE_".
        """

        output = triton_to_np_dtype(api_type.split("TYPE_")[1])
        if output is None:
            raise ValueError(f"Unrecognized type '{str(api_type)}'")
        return output

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

        Notes
        -----
        All types listed in model configurations are prepended with "TYPE_".
        """

        output = np_to_triton_dtype(dtype)
        if output is None:
            raise ValueError(f"Unrecognized type '{str(dtype)}'")
        return output

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
        inputs : numpy.ndarray or dict of numpy.ndarray
            For single-input models provide a single numpy array. Multi-input
            models require a dict that keys input names to numpy arrays. This
            dict is required because the ordering of model inputs appearing in
            `model_dict` is not reliable.
        model_dict : dict
            A dictionary describing the name, shape, and type of inputs and
            outputs, and the maximum batch size if defined.
        strict_types : bool
            Enforce strict types on model inputs. _client_inputs will
            cast inputs to the correct type, so this is not necessary.
            Default value is False.

        See also
        --------
        model_metadata

        Notes
        -----
        For multi-input models, secondary inputs such as parameters require a
        batch dimension. This must be equal to the data batch size and so these
        parameters must be repeated.
        """

        def _compare_types(model_dict, provided, expected):
            if self._np_to_api_types(provided.dtype) != expected["dataType"]:
                raise Exception(
                    (
                        f"Model {model_dict['name']} input "
                        f"{expected['name']} "
                        f"expects type {expected['dataType']}, received input with "
                        f"type {provided.dtype}."
                    )
                )

        def _compare_dims(provided, expected):
            return all(
                [True if e == -1 else p == e for (p, e) in zip(provided, expected)]
            )

        def _int(shape):
            return [int(s) for s in shape]

        def _compare_shape(model_dict, provided, expected, batching):
            if batching:
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

        # validate type and number of inputs
        if isinstance(inputs, ARRAY_TYPES):
            if len(model_dict["input"]) != 1:
                raise Exception(
                    (
                        f"Model {model_dict['name']} with {len(model_dict['input'])}"
                        " inputs requires dictionary of numpy arrays, keyed to"
                        " input names."
                    )
                )
        elif isinstance(inputs, dict):
            if len(inputs) != len(model_dict["input"]):
                raise Exception(
                    (
                        f"Model {model_dict['name']} expects {len(model_dict['input'])}"
                        f" inputs, received {len(inputs)}."
                    )
                )
        else:
            raise Exception(
                (
                    "inputs must be an np.ndarray or a dictionary of np.ndarrays"
                    " keyed to the model inputs."
                )
            )

        # verify that all inputs are found for dict input
        if isinstance(inputs, dict):
            for i in model_dict["input"]:
                if i["name"] not in inputs.keys():
                    raise Exception(f"Model input {i['name']} not found in inputs.")

        # if model is batching -> batch dimension implied
        if "maxBatchSize" in model_dict:
            if model_dict["maxBatchSize"] > 0:
                batching = True
                if isinstance(inputs, ARRAY_TYPES):
                    batch_size = inputs.shape[0]
                else:
                    batch_size = inputs[list(inputs.keys())[0]].shape[0]
            else:
                batching = False
        else:
            batching = False

        # validate input types
        if strict_types:
            if isinstance(inputs, ARRAY_TYPES):
                _compare_types(model_dict, inputs, model_dict["input"][0])
            else:
                for i in model_dict["input"]:
                    _compare_types(model_dict, inputs[i["name"]], i)

        # validate input shapes
        if isinstance(inputs, ARRAY_TYPES):
            _compare_shape(model_dict, inputs, model_dict["input"][0], batching)
        else:
            for i in model_dict["input"]:
                _compare_shape(model_dict, inputs[i["name"]], i, batching)

        # validate batching
        if batching:
            if isinstance(inputs, ARRAY_TYPES):
                if inputs.shape[0] > model_dict["maxBatchSize"]:
                    raise Exception(
                        (
                            f"Model {model_dict['name']} max batch size "
                            f"{model_dict['maxBatchSize']} exceeded."
                        )
                    )
            else:
                for i in inputs.values():
                    if i.shape[0] > model_dict["maxBatchSize"]:
                        raise Exception(
                            (
                                f"Model {model_dict['name']} max batch size "
                                f"{model_dict['maxBatchSize']} exceeded."
                            )
                        )
                    if i.shape[0] != batch_size:
                        raise Exception(
                            (
                                "Non-uniform batch size of inputs. Expected"
                                f" {batch_size}, found {i.shape[0]}."
                            )
                        )

    def _client_inputs(self, inputs, model_dict):
        """Generates tritonclient.grpc.InferInput objects for client.

        Given inputs and a model configuration and metadata, this function
        validates the inputs against the model expectations, and generates the
        InferInput objects used by the client to transmit inputs to triton.

        Parameters
        ----------
        inputs : numpy.ndarray or dict of numpy.ndarray
            For single-input models provide a single numpy array. Multi-input
            models require a dict that keys input names to numpy arrays. This
            dict is required because the ordering of model inputs appearing in
            `model_dict` is not reliable.
        model_dict : dict
            A dictionary describing the name, shape, and type of inputs and
            outputs, as well as maximum batch size.

        Outputs
        -------
        infer_inputs : list of tritonclient.grpc.InferInput
            A list of InferInput objects populated with data.
        """

        # validate inputs against model expectations
        self._validate_inputs(inputs, model_dict)

        # create InferInput objects for each model input
        import tritonclient.grpc as grpcclient

        def add_input(provided, expected):
            iio = grpcclient.InferInput(
                expected["name"], provided.shape, expected["dataType"].split("TYPE_")[1]
            )
            iio.set_data_from_numpy(provided)
            return iio

        if isinstance(inputs, np.ndarray):
            infer_inputs = [add_input(inputs, model_dict["input"][0])]
        elif isinstance(inputs, SharedNumpyArray):
            infer_inputs = [add_input(inputs.view(), model_dict["input"][0])]
        else:
            infer_inputs = [
                add_input(
                    (
                        inputs[i["name"]]
                        if isinstance(inputs[i["name"]], np.ndarray)
                        else inputs[i["name"]].view()
                    ),
                    i,
                )
                for i in model_dict["input"]
            ]
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

        # create InferRequestedOutput objects for each model output
        import tritonclient.grpc as grpcclient

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

                    # clear result
                    request["result"] = []

                    # inference generated an exception
                    if type(results) == InferenceServerException:

                        # capture error in request
                        if request["attempts"] == 1:
                            request["errors"] = []
                        request["errors"].append(results.message())

                        # make another attempt if retry limit has not been reached
                        if request["attempts"] < self.retries:
                            retry.append(request)
                        else:  # indicate failure and add request to output list
                            request["success"] = False
                            completed.append(request)

                    # inference generated a result
                    else:

                        # convert responses to numpy arrays
                        for j, output in enumerate(
                            self.model_dicts[request["model_name"]]["output"]
                        ):
                            request["result"].append(results.as_numpy(output["name"]))

                        # add request to output list
                        request["success"] = True
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
            model = TritonModel(model_name, self.url)
            model_dict = model.get_config()
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


import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=("Testing nargs."))
    parser.add_argument(
        "input",
        type=str,
        nargs="*",
        help="Path to file or file pattern.",
    )
    parser.add_argument("output", type=str, help="Output directory.")
    parser.add_argument(
        "-f",
        "--files",
        required=False,
        type=str,
        help="Path to root folder containing images, or text file listing image paths.",
    )
    parser.add_argument(
        "-n",
        "--noskip",
        dest="skip",
        action="store_false",
        help=("Overwrite existing images (non-default)."),
    )
    parser.add_argument(
        "-s",
        "--server",
        required=False,
        default="localhost:8001",
        type=str,
        help="Triton server address. Default value is `localhost:8001`.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "-t",
        "--tile",
        required=False,
        default=None,
        type=int,
        help="Tile size. Defaults to internal file tile size at scan magnification.",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        required=False,
        default=0,
        type=int,
        help="Tile overlap. Defaults to 0 pixels.",
    )
    parser.add_argument(
        "-M",
        "--magnification",
        required=False,
        default=None,
        type=float,
        help="Magnification. Defaults to scan magnification.",
    )
    parser.add_argument(
        "-i",
        "--icc",
        action="store_true",
        help="Apply ICC correction. Defaults to False.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        required=False,
        default=128,
        type=int,
        help=("Batch size. Defaults 128 tiles."),
    )
    parser.add_argument(
        "-p",
        "--prefetch",
        required=False,
        default=4,
        type=int,
        help=("The number of prefetch batches."),
    )
    parser.add_argument(
        "-w",
        "--workers",
        required=False,
        default=32,
        type=int,
        help=("The number of loader processes (default 32)."),
    )
    parser.add_argument(
        "-ma",
        "--mask_dir",
        required=True,
        default=None,
        type=str,
        help=("The directory where the masks are stored"),
    )
    args = parser.parse_args()

    # parse inputs - pattern expansion or file containing list of files
    if isinstance(args.input, list):
        files = [os.path.join(os.getcwd(), f) for f in args.input]
    elif os.path.isfile(args.files):
        with open(args.files) as f:
            files = [line.split()[0] for line in f]
    else:
        raise ValueError(
            "Provide one of a text file containing inputs via -f/--files or a file pattern."
        )

    # throw an error when mask directory has not been inputted
    if args.mask_dir is None:
        raise ValueError("Provide the directory where the masks have been stored.")

    # check of model is loaded
    model = TritonModel(args.model, args.server)
    if model.is_loaded:
        print("The model has already been loaded")
    else:
        try:
            # load tensorflow model - set maximum batch size
            model = TritonModel(args.model, args.server)

            # initialize builder with a basic configuration
            builder = ConfigBuilder(args.model, config={"maxBatchSize": args.batch})

            # increase the number of model instances per GPU to 2
            builder.add_instance_group(count=2)

            # add automatic mixed precision
            builder.add_mixed_precision()

            # re-load model with new config
            model.load(config=builder.config)

            assert model.is_loaded()
        except Exception as e:
            print("loading model failed: " + str(e), flush=True)

    # iterate through files
    for file in files:
        # determine magnification, tile size if not provided
        source = large_image_source_tiff.open(file)
        metadata = source.getMetadata()
        magnification = (
            metadata["magnification"]
            if args.magnification is None
            else args.magnification
        )
        t = (
            (metadata["tileHeight"], metadata["tileWidth"])
            if args.tile is None
            else (args.tile, args.tile)
        )
        if (args.tile is None) and (magnification != metadata["magnification"]):
            raise ValueError(
                (
                    "Using default tile size requires native magnification "
                    f"`None` or {metadata['magnification']}."
                )
            )

        # create tile source
        mask_path = os.path.join(
            args.mask_dir, file.split("/")[-1].replace(".svs", ".svs_1.25_mask.png")
        )
        hs_study = study(
            (file, mask_path),
            t=(t, t),
            chunk=(t, t),
            objective=magnification,
            mask_threshold=0.5,
        )

        # tile iterator
        iterator = TiffPrefetch(
            hs_study, np.uint8, args.icc, args.batch, args.prefetch, args.workers
        )

        # inference
        features, metadata, times, failures = inference(
            iterator, args.model, url=args.server, limit=1, rest=0.0
        )

        # concatenate features
        features = np.concatenate(features[0], axis=0)

        # write to tfrecord
        tfr_file = "{}/{}.{}_{}_{}X.tfr".format(
            args.output, file.split("/")[-1], args.model, t, args.overlap, magnification
        )
        write_record(
            tfr_file,
            features,
            metadata,
            labels={},
            structured=False,
            precision=tf.float16,
        )


if __name__ == "__main__":
    main()
