from google.protobuf import json_format, text_format
import numpy as np
import os
from tritonclient.utils import np_to_triton_dtype
from tritonclient.grpc import model_config_pb2


ONNX_TRT_WHITELIST = {"precision_mode", "max_workspace_size_bytes"}


class DynamicBatching(object):
    """Dynamic batching configuration.

    Dynamic batching allows the aggregation of multiple requests into a single
    inference for optimization.

    Parameters
    ----------
    preferred_batch_size : list
        A list of one or more preferred batch sizes. Default value is [64].
    max_queue_delay_microseconds : int
        The maximimum wait time for dynamic batching. After expiration a request will proceed even
        if the aggregated requests do not meet the preferred batch size. Default value is 0.
    preserve_ordering : bool
        Preserve the order of batches as they are received. Default value is True.

    References
    ----------
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
    """

    def __init__(
        self,
        preferred_batch_size=None,
        max_queue_delay_microseconds=0,
        preserve_ordering=True,
    ):
        if preferred_batch_size is None:
            preferred_batch_size = [64]
        self.config = {
            "PreferredBatchSize": preferred_batch_size,
            "max_queue_delay_microseconds": max_queue_delay_microseconds,
            "preserve_ordering": preserve_ordering,
        }


class ModelInput(object):
    """Model input configuration.

    Configuration of one model input. Pass a list of these inputs for a multi-input model.

    Parameters
    ----------
    name : str
        Model input name.
    shape : list or tuple of int
        The shape of the input not including batch dimension. Variable dimensions are set to -1.
    dtype : numpy.dtype
        The numpy dtype of the model input.
    optional : bool
        Whether this input is optional. Default value is False.
    """

    def __init__(self, name, shape, dtype, optional=False):
        if not isinstance(name, str):
            raise ValueError("name must be str")
        if not isinstance(shape, (list, tuple, np.ndarray)):
            raise ValueError("shape must be type list or type np.ndarray of type int")
        if not all([isinstance(i, (int, np.integer)) for i in shape]):
            raise ValueError("elements of shape must be type int")
        dtype = f"TYPE_{np_to_triton_dtype(dtype().dtype)}"
        input = {
            "name": name,
            "dataType": dtype,
            "dims": shape,
        }
        if optional:
            input["optional"] = True
        self.config = input


class ModelOutput(ModelInput):
    """Model output configuration.

    Configuration of one model output. Pass a list of these inputs for a multi-output model.

    Parameters
    ----------
    name : str
        Model output name.
    shape : list or tuple of int
        The shape of the input not including batch dimension. Variable dimensions are set to -1.
    dtype : numpy.dtype
        The numpy dtype of the model input.
    """

    def __init__(self, name, shape, dtype):
        super().__init__(name, shape, dtype, False)


class InstanceGroup(object):
    """Instance group configuration.

    Configures the number and type (CPU/GPU) of processors and the number of model instances
    hosted on each.

    Parameters
    ----------
    count : int
        The number of model instances to run concurrently. Default value is 1.
    kind : str {"cpu", "gpu"}
        Default value of "gpu" specifies that `count` models be hosted on each
        available gpu. Default value is `gpu`.
    gpus : list of int
        If specified, `count` instances will be hosted on each of the listed
        gpus. For example, [0, 1] would specify serving on gpus zero and one.
        Default value of `None` means that `count` instances will be served on
        each available gpu.

    References
    ----------
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups
    """

    def __init__(self, count=1, kind="gpu", gpus=None):
        if not isinstance(count, int):
            raise ValueError("`count` must be int.")
        if kind.lower() not in {"cpu", "kind_cpu", "gpu", "kind_gpu"}:
            raise ValueError("`kind` must be one of 'cpu', 'gpu'.")
        elif kind in {"cpu", "kind_cpu"}:
            kind = "KIND_CPU"
        else:
            kind = "KIND_GPU"
        if gpus is not None:
            if not isinstance(gpus, list):
                raise ValueError("argument 'gpus' must be list of type int.")
            if not all([isinstance(inst, int) for inst in gpus]):
                raise ValueError("elements of 'gpus' must be type int.")

        # set count, kind, and optionally GPUs
        instance = {"count": count, "kind": kind}
        if gpus is not None:
            instance["gpus"] = gpus
        self.config = instance


class PythonOptimization(object):
    """Python backend optimization configuration.

    For the python backend page locking of memory used in host-device transfer is the only
    optimization avaialble.

    Parameters
    ----------
    input_pinned : bool
        Page lock memory used to send model inputs. Default value is True.
    output_pinned : bool
        Page lock memory used to recieve model outputs. Default value is True.

    References
    ----------
    https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/#pinned_host_memory
    """

    def __init__(self, input_pinned=True, output_pinned=True):
        self.config = {
            "inputPinnedMemory": {"enable": input_pinned},
            "outputPinnedMemory": {"enable": output_pinned},
        }


class PythonConfig(object):
    """A model configuration for the python backend.

    This class can generate JSON format dictionaries for use with model loading
    functions, and can save configurations in protocol buffer format for
    file-based configuration.

    Parameters
    ----------
    name : str
        Model name as stored in the model repository.
    max_batch_size : int
        The maximum number of samples in a request. Use 0 for a non-batching model.
    input : ModelInput or list
        Model inputs.
    output : ModelOutput or list
        Model outputs.
    instance_group : InstanceGroup
        An instance group configuration defining model resources.
    optimization : PythonOptimization
        Python backend optimization configuration. Default value None enables
        pinned memory by default.
    response_cache : bool
        Whether to cache model input-output pairs. See reference below. Default value
        is False for no caching.

    References
    ----------
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/response_cache.md
    """

    def __init__(
        self,
        name,
        max_batch_size,
        input=None,
        output=None,
        instance_group=None,
        dynamic_batching=None,
        optimization=None,
        response_cache=False,
    ):
        if not isinstance(name, str):
            raise ValueError("`name` must be type str.")
        if not isinstance(max_batch_size, int):
            raise ValueError("`max_batch_size` must be type int.")
        if input is not None:
            if not isinstance(input, (ModelInput, list)):
                raise ValueError(
                    "`input` must be a ModelInput object or a list of ModelInput objects."
                )
            if isinstance(input, list):
                if not all([isinstance(i, (ModelInput)) for i in input]):
                    raise ValueError("elements of `input` must be a ModelInput object.")
        if output is not None:
            if not isinstance(output, (ModelOutput, list)):
                raise ValueError(
                    "`output` must be a ModelOutput object or a list of ModelOutput objects."
                )
            if isinstance(output, list):
                if not all([isinstance(i, (ModelOutput)) for i in output]):
                    raise ValueError(
                        "elements of `output` must be a ModelOutput object."
                    )
        if instance_group is not None:
            if not isinstance(instance_group, InstanceGroup):
                raise ValueError("`instance_group` must be an InstanceGroup object.")
        if not isinstance(response_cache, bool):
            raise ValueError("`response_cache` must be type bool.")
        self.config = {
            "name": name,
            "versionPolicy": {"latest": {"numVersions": 1}},
            "maxBatchSize": max_batch_size,
            "responseCache": {"enable": response_cache},
            "backend": "python",
        }
        if input is not None:
            self.config["input"] = (
                [i.config for i in input] if input is list else [input.config]
            )
        if output is not None:
            self.config["output"] = (
                [o.config for o in output] if output is list else [output.config]
            )
        if instance_group is not None:
            self.config["instanceGroup"] = [instance_group.config]
        if optimization is not None:
            print(optimization)
            self.config["optimization"] = optimization.config

    def json(self):
        """Return the python model configuration as a JSON dictionary.

        The JSON dictionary can be used with model loading functions.
        """

        return self.config

    def protobuffer(self):
        """Return the python model configuratoin as a protocol buffer."""

        return json_format.ParseDict(self.json(), model_config_pb2.ModelConfig())

    def save(self, path):
        """Save the configuration in protobuffer text (config.pbtxt) format.

        Parameters
        ----------
        path : str
            Path for the output file. File naming is automatic.
        """

        message = json_format.ParseDict(self.json(), model_config_pb2.ModelConfig())
        if not os.path.isdir(path):
            raise ValueError(f"`path` {path} does not exist.")
        with open(os.path.join(path, "config.pbtxt"), "w") as output:
            text_format.PrintMessage(message, output)


class TensorRt(object):
    """TensorRT configuration.

    For use with the TensorFlow and ONNX backends.

    Parameters
    ----------
    precision_mode : str {FP16, FP32}
        Model precision either half-float (FP16) or float (FP32). Default value is "FP16".
    max_cached_engines : int
        The maximum cached TensorRT engines in TensorRT operations. Default value is 100.
    minimum_segment_size : int
        The smallest subgraph size considered for TensorRT optimization. Default value is 3.
    max_workspace_size : int
        The maximum GPU memory available during model execution. Default value is 4 GB.

    References
    ----------
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#onnx-with-tensorrt-optimization-ort-trt
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#tensorflow-with-tensorrt-optimization-tf-trt
    https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    def __init__(
        self,
        precision_mode="FP16",
        max_cached_engines=100,
        minimum_segment_size=3,
        max_workspace_size_bytes=4294967296,
    ):
        if precision_mode.upper() not in {"FP16", "FP32"}:
            raise ValueError("`precision_mode` must be one of 'FP16' or 'FP32'.")
        if not isinstance(max_cached_engines, int):
            raise ValueError("`max_cached_engines` must be type int.")
        if not isinstance(minimum_segment_size, int):
            raise ValueError("`minimum_segment_size` must be type int.")
        if not isinstance(max_workspace_size_bytes, int):
            raise ValueError("`max_workspace_size_bytes` must be type int.")
        self.config = {
            "executionAccelerators": {
                "gpuExecutionAccelerator": [
                    {
                        "name": "tensorrt",
                        "parameters": {
                            "precision_mode": f"{precision_mode.upper()}",
                            "max_cached_engines": str(max_cached_engines),
                            "minimum_segment_size": str(minimum_segment_size),
                            "max_workspace_size_bytes": str(max_workspace_size_bytes),
                        },
                    }
                ],
            },
        }


class TensorflowXla(object):
    """Tensorflow XLA graph optimization.

    Sets the level of XLA just in time compilation of tensorflow models.

    Parameters
    ----------
    level : int {-1, 0, 1, 2}
        The optimization level can be off (-1), off but delayed (0), moderate
        optimization(1), or higher optimization (2). Default value is 2.
    """

    def __init__(self, level=2):
        if not isinstance(level, int) or not level in {-1, 0, 1, 2}:
            raise ValueError("`level` must be type int with of -1, 0, 1, or 2.")
        self.config = {"graph": {"level": level}}


class TensorflowMixedPrecision(object):
    """Tensorflow automatic mixed precision configuration.

    A configuration that activates automatic mixed precision for half-float inference.
    """

    def __init__(self):
        self.config = {
            "executionAccelerators": {
                "gpuExecutionAccelerator": [{"name": "auto_mixed_precision"}]
            }
        }


class TensorflowOptimization(PythonOptimization):
    """Tensorflow backend optimization configuration.

    The Tensorflow backend supports the memory page locking as well as mixed precision,
    tensorrt, and xla compilation. The default optimization enables page locking and
    mixed precision.

    Parameters
    ----------
    input_pinned : bool
        Page lock memory used to send model inputs. Default value is True.
    output_pinned : bool
        Page lock memory used to recieve model outputs. Default value is True.
    amp : TensorflowMixedPrecision
        An TensorflowMixedPrecision configuration to enable half-float inference.
        Default value is None.
    trt : TensorRt
        A TensorRt configuration to enable reduced precision and operation fusion.
        Default value is None.
    xla : TensorflowXLA
        A TensorflowXla configuration for XLA jit compilation.
        Default value is None.

    References
    ----------
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#tensorflow-with-tensorrt-optimization-tf-trt
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#tensorflow-automatic-fp16-optimization
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#tensorflow-jit-graph-optimizations
    """

    def __init__(
        self,
        input_pinned=True,
        output_pinned=True,
        amp=None,
        trt=None,
        xla=None,
    ):
        super(TensorflowOptimization, self).__init__(input_pinned, output_pinned)
        if amp is not None and trt is not None:
            raise ValueError("`trt` cannot be enabled concurrently with `amp`.")
        if amp is not None:
            if not isinstance(amp, TensorflowMixedPrecision):
                raise ValueError("`amp` must be a TensorflowMixedPrecision object.")
            self.config.update(amp.config)
        if trt is not None:
            if not isinstance(trt, TensorRt):
                raise ValueError("`trt` must be a TensorRt object.")
            self.config.update(trt.config)
        if xla is not None:
            if not isinstance(xla, TensorflowXla):
                raise ValueError("`xla` must be a TensorflowXla object.")
            self.config.update(xla.config)


class TensorflowConfig(PythonConfig):
    """A model configuration for the tensorflow backend.

    This class can generate JSON format dictionaries for use with model loading
    functions, and can save configurations in protocol buffer format for
    file-based configuration.

    Parameters
    ----------
    name : str
        Model name as stored in the model repository.
    input : ModelInput or list
        Model inputs.
    output : ModelOutput or list
        Model outputs.
    instance_group : InstanceGroup
        An instance group configuration defining model resources.
    max_batch_size : int
        The maximum number of samples in a request. Use 0 for a non-batching model.
    optimization : TensorflowOptimization
        Python backend optimization configuration. Default value None enables
        pinned memory by default.
    response_cache : bool
        Whether to cache model input-output pairs. See reference below. Default value
        is False for no caching.

    References
    ----------
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/response_cache.md
    """

    def __init__(
        self,
        name,
        max_batch_size,
        input=None,
        output=None,
        instance_group=None,
        dynamic_batching=None,
        optimization=None,
        response_cache=False,
    ):
        super(TensorflowConfig, self).__init__(
            name=name,
            input=input,
            output=output,
            instance_group=instance_group,
            max_batch_size=max_batch_size,
            dynamic_batching=dynamic_batching,
            response_cache=response_cache,
        )
        if optimization is not None:
            if not isinstance(optimization, TensorflowOptimization):
                raise ValueError(
                    "`optimization` must be a TensorflowOptimization object."
                )
            self.config["optimization"] = optimization.config
        self.config["backend"] = "tensorflow"
        self.config["platform"] = "tensorflow_savedmodel"


class OnnxGraph(object):
    """ONNX graph optimization.

    Sets the level of graph optimization for Onnx models.

    Parameters
    ----------
    level : int {-1, 1, 2}
        The optimization level can be basic (-1), extended (1), or disabled (2).

    References
    ----------
    https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
    """

    def __init__(self, level=1):
        if not isinstance(level, int) or not level in {-1, 1, 2}:
            raise ValueError("`level` must be type int with of -1, 1, or 2.")
        self.config = {"graph": {"level": level}}


class OnnxOptimization(PythonOptimization):
    """ONNX backend optimization configuration.

    The ONNX backend supports the memory page locking and tensorrt optimizations.
    The default optimization enables page locking. Openvino optmization is currently
    not supported.

    Parameters
    ----------
    input_pinned : bool
        Page lock memory used to send model inputs. Default value is True.
    output_pinned : bool
        Page lock memory used to recieve model outputs. Default value is True.
    trt : TensorRt
        A TensorRt configuration to enable reduced precision and operation fusion.
        Default value is None.
    graph : OnnxGraph
        An OnnxGraph configuration to enable graph optimizations including redundant
        operation elimination and operation fusion.

    References
    ----------
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html#onnx-with-tensorrt-optimization-ort-trt
    """

    def __init__(
        self,
        input_pinned=True,
        output_pinned=True,
        trt=None,
        graph=None,
    ):
        super(OnnxOptimization, self).__init__(input_pinned, output_pinned)
        if graph is not None and trt is not None:
            raise ValueError("`trt` cannot be enabled concurrently with `graph`.")
        if trt is not None:
            if not isinstance(trt, TensorRt):
                raise ValueError("`trt` must be a TensorRt object.")
            parameters = trt.config["executionAccelerators"]["gpuExecutionAccelerator"][
                0
            ]["parameters"]
            parameters = {
                k: v for k, v in parameters.items() if k in ONNX_TRT_WHITELIST
            }
            trt.config["executionAccelerators"]["gpuExecutionAccelerator"][0][
                "parameters"
            ] = parameters
            self.config.update(trt.config)
        if graph is not None:
            if not isinstance(graph, OnnxGraph):
                raise ValueError("`graph` must be a OnnxGraph object.")
            self.config.update(graph.config)


class OnnxConfig(PythonConfig):
    """A model configuration for the ONNX backend.

    This class can generate JSON format dictionaries for use with model loading
    functions, and can save configurations in protocol buffer format for
    file-based configuration.

    Parameters
    ----------
    name : str
        Model name as stored in the model repository.
    input : ModelInput or list
        Model inputs.
    output : ModelOutput or list
        Model outputs.
    instance_group : InstanceGroup
        An instance group configuration defining model resources.
    max_batch_size : int
        The maximum number of samples in a request. Use 0 for a non-batching model.
    optimization : TensorflowOptimization
        Python backend optimization configuration. Default value None enables
        pinned memory by default.
    response_cache : bool
        Whether to cache model input-output pairs. See reference below. Default value
        is False for no caching.

    References
    ----------
    https://github.com/triton-inference-server/server/blob/main/docs/user_guide/response_cache.md
    """

    def __init__(
        self,
        name,
        max_batch_size,
        input=None,
        output=None,
        instance_group=None,
        dynamic_batching=None,
        optimization=None,
        response_cache=False,
    ):
        super(OnnxConfig, self).__init__(
            name=name,
            input=input,
            output=output,
            instance_group=instance_group,
            max_batch_size=max_batch_size,
            dynamic_batching=dynamic_batching,
            response_cache=response_cache,
        )
        if optimization is not None:
            if not isinstance(optimization, OnnxOptimization):
                raise ValueError("`optimization` must be an OnnxOptimization object.")
            self.config["optimization"] = optimization.config
        self.config["backend"] = "onnxruntime"


def load(path):
    """Load a JSON dictionary configuration from a protobuffer text file.

    Parameters
    ----------
    path : str
        Path for the input file.
    """

    with open(path, "rb") as f:
        protobuf = text_format.Parse(f.read(), model_config_pb2.ModelConfig())
    return json_format.MessageToDict(protobuf)
