from simple_triton.model import model_config


class ConfigBuilder(object):
    """Build a configuration for a Triton hosted model.

    The model configuration defines hosting resources like GPUs and CPUs,
    number of model instances per resource, model input and output names,
    shapes, and type, optimizations like TensorRT and Automatic Mixed
    Precision, and dynamic batching and queuing preferences.

    Parameters
    ----------
    config : dict
        An initial configuration. If `None`, a client and name of a
        hosted model must be provided to obtain an initial configuration.
        Default value is `None`.
    client : tritonclient.grpc.InferenceServerClient
        A remote-procedure call client for the triton server. If `config`
        is `None`, this client must be provided to query a config from
        the server. Default value is `None`.
    model_name : string
        The name of the model to query as hosted in triton. Must be
        provided if `config` is None.

    Attributes
    ----------
    config : dict
        A model configuration. Can be used to alter the configuration of
        a hosted model by reloading.

    Methods
    -------
    response_cache(enable=False)
        Sets caching of inference results on-server.
    add_input(name, datatype, dims)
        Add or modifies a model input given input name and dims.
    max_batch_size(samples)
        Set the maximum batch size.
    add_instance_group(count=1, kind="gpu", gpus=None)
        Add an instance group defining the model hardware resources and
        instances.
    remove_instance_groups()
        Remove all instance groups and default to 1 instance per GPU.
    add_mixed_precision()
        Enable automatic mixed precision for half-float operations.
    remove_mixed_precision
        Disable automatic mixed precision for half-float operations.
    add_trt(precision="FP16")
        Enable TensorRT optimization.
    remove_trt(self, precision="FP16")
        Disable TensorRT optimization.

    Notes
    -----
    This class does not build a configuration from scratch. It is
    typically used to modify an auto-generated configuration produced
    by Triton when a model is loaded. When launching Triton we
    reccomend using `--model-control-mode=explicit` to enable the
    configuration of loaded models to be modified without restart, and
    `--strict-model-config=false` which relaxes the requirements of a
    valid configuration, so that not every setting has to be provided
    by the client or user.

    References
    ----------
    The ModelConfig protobuf https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    def __init__(self, config=None, client=None, model_name=None):
        """Initialize from provided config or as hosted. Disable
        response cache by default
        """
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("config must be a dict")
            self.config = config
        else:
            self.config = model_config(client, model_name)["config"]
        self.response_cache(False)

    def _gpu_accelerator_status(self, accelerator):
        """Test if a gpuExecutionAccelerator is present."""

        if "optimization" in self.config:
            if "executionAccelerators" in self.config["optimization"]:
                if (
                    "gpuExecutionAccelerator"
                    in self.config["optimization"]["executionAccelerators"]
                ):
                    for d in self.config["optimization"]["executionAccelerators"][
                        "gpuExecutionAccelerator"
                    ]:
                        if "name" in d:
                            if d["name"] == accelerator:
                                return True
                    return False
                else:
                    return False
            else:
                return False
        else:
            return False

    def _gpu_accelerator_delete(self, accelerator):
        """Delete a gpuExecutionAccelerator and cleanup empty parents"""

        if gpu_accelerator_status(self.config, accelerator):
            gpuexecacc = self.config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ]
            keep = []
            for i, d in enumerate(gpuexecacc):
                if "name" in enumerate(d):
                    if d["name"] != accelerator:
                        keep.append(i)
            self.config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ] = [gpuexecacc[i] for i in keep]
            if (
                len(
                    self.config["optimization"]["executionAccelerators"][
                        "gpuExecutionAccelerator"
                    ]
                )
                == 0
            ):
                del self.config["optimization"]["executionAccelerators"][
                    "gpuExecutionAccelerator"
                ]
            if self.config["optimization"]["executionAccelerators"] == {}:
                del self.config["optimization"]["executionAccelerators"]
            if self.config["optimization"] == {}:
                del self.config["optimization"]

    def _gpu_accelerator_add(self, accelerator):
        """Adds a gpu accelerator to the config"""
        if gpu_accelerator_status(self.config, accelerator["name"]):
            gpu_accelerator_delete(self.config, accelerator["name"])
        if "optimization" not in self.config:
            self.config["optimization"] = {}
        if "executionAccelerators" not in self.config["optimization"]:
            self.config["optimization"]["executionAccelerators"] = {}
        if (
            "gpuExecutionAccelerator"
            not in self.config["optimization"]["executionAccelerators"]
        ):
            self.config["optimization"]["executionAccelerators"][
                "gpuExecutionAccelerator"
            ] = []
        self.config["optimization"]["executionAccelerators"][
            "gpuExecutionAccelerator"
        ].append(accelerator)

    def _gpu_accelerator_trt(precision="FP16"):
        """Returns a TensorRT accelerator"""
        if precision.upper() == "FP16" or precision.upper() == "FP32":
            return {
                "name": "tensorrt",
                "parameters": {"precision_mode": f"{precision.upper()}"},
            }
        else:
            raise ValueError(
                "precision must be one of None, numpy.float16, numpy.float32"
            )

    def _gpu_accelerator_amp():
        """Returns an amp acccelerator"""
        return {"name": "auto_mixed_precision"}

    def response_cache(self, enable=False):
        """Set model response cache.

        When enabled, the result of prior inferences will be cached and returned
        if the same input is received. This can intefere with benchmarking and is
        disabled by default.

        Parameters
        ----------
        enable : bool
            If `True` the response cache will be enabled. Default value is `False`.
        """

        self.config["response_cache"] = {"enable": enable}

    def add_input(self, name, datatype, dims):
        """Add a new input or set the properties of an existing input.

        Parameters
        ----------
        name : str
            The input name. Used to check if the input already exists. And existing
            input will be overwritten while a new input will be added.
        datatype : str
            A valid triton datatype (see below).
        dims : array-like
            A list of integers indicating the model input sizes sans batch. Variable
            sizes should be indicated with a `-1`.

        Notes
        -----
        See Triton for valid `datatype` values:
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
        """

        # check for valid inputs
        if not isinstance(name, str):
            raise ValueError("name must be str")
        if not isinstance(name, str):
            raise ValueError("datatype must be str")
        if not isinstance(dims, (list, np.ndarray)):
            raise ValueError("dims must be a list or np.ndarray of int")
        if not all([isinstance(i, (int, np.integer)) for i in dims]):
            raise ValueError("elements of dims must be int")

        inputs = [i["name"] for i in self.config["input"]]
        if name in inputs:
            index = inputs.index(name)
            self.config["inputs"][index]["datatype"] = datatype
            self.config["inputs"][index]["dims"] = dims
        else:
            inputs = {
                "name": name,
            }
            self.config["inputs"].append(
                {"name": name, "datatype": datatype, "dims": dims}
            )

    def max_batch_size(self, samples):
        """Sets the maximum batch size for the config.

        Batch sizes exceeding this value will trigger an inference exception.

        Parameters
        ----------
        samples : int
            The maximum batch size for the model.
        """

        if not isinstance(samples, int):
            raise ValueError("argument 'samples' must be int.")
        self.config["maxBatchSize"] = samples

    def add_instance_group(self, count=1, kind="gpu", gpus=None):
        """Adds an instance group to the config.

        The instance group specifies the number of model instances hosted on each
        CPU and GPU. By default, 1 model instance will be hosted on each available
        GPU. Specific GPUs can be set using the `gpus` argument. Each call adds to
        the existing instance group specification.

        Parameters
        ----------
        count : int
            The number of model instances to run concurrently.
        kind : str {"cpu", "gpu"}
            Default value of "gpu" specifies that `count` models be hosted on each
            available gpu.
        gpus : list of int
            If specified, `count` instances will be hosted on each of the listed
            gpus. For example, [0, 1] would specify serving on gpus zero and one.
            Default value of `None` means that `count` instances will be served on
            each available gpu.

        References
        ----------
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups
        """

        # check input validity
        if not isinstance(count, int):
            raise ValueError("argument 'count' must be int.")
        if kind.lower() not in ["cpu", "kind_cpu", "gpu", "kind_gpu"]:
            raise ValueError("argument 'kind' must be one of 'cpu', 'gpu'.")
        elif kind in {"cpu", "kind_cpu"}:
            kind = "KIND_CPU"
        else:
            kind = "KIND_GPU"
        if gpus is not None:
            if not isinstance(gpus, list):
                raise ValueError("argument 'gpus' must be list of int.")
            if not all([isinstance(inst, int) for inst in gpus]):
                raise ValueError("elements of 'gpus' must be int.")

        # set model name, count, and kind
        instance = {"name": self.config["name"], "count": count, "kind": kind}
        if gpus is not None:
            instance["gpus"] = gpus

        # assign
        if "instanceGroup" not in self.config:
            config["instanceGroup"] = [instance]
        else:
            config["instanceGroup"].append(instance)

    def remove_instance_groups(self):
        """Removes all instance groups from config."""
        if "instanceGroup" in self.config:
            del config["instanceGroup"]

    def add_mixed_precision(self):
        """Add an automatic mixed-precision accelerator to the config.

        This enables the model to perform half-precision operations to
        accelerate inference and decrease memory consumption. Cannot be
        used simultaneously with TensorRT accelerator.
        """

        if self._gpu_accelerator_status(self.config, "tensorrt"):
            self._gpu_accelerator_delete(self.config, "tensorrt")
        self._gpu_accelerator_add(self.config, _gpu_accelerator_amp())

    def remove_mixed_precision(self):
        """Remove an automatic mixed-precision accelerator from the config."""
        if self._gpu_accelerator_status(self.config, "auto_mixed_precision"):
            self._gpu_accelerator_delete(self.config, "auto_mixed_precision")

    def add_trt(self, precision="FP16"):
        """Add an TensorRT accelerator to the config.

        This analyzes the model using TensorRT (TRT) to optimize inference
        via quantization, layer and tensor fusion, and kernel tuning.
        TRT can be applied to produce either an FP32 or FP16 model.

        Parameters
        ----------
        precision : str {"FP16", "FP32"}
            Precision to use when applying TensorRT. Default value is `"FP16"`.
        """

        if precision.upper() not in {"FP16", "FP32"}:
            raise ValueError("precision must be one of 'FP16', 'FP32'.")
        if self._gpu_accelerator_status(self.config, "auto_mixed_precision"):
            self._gpu_accelerator_delete(self.config, "auto_mixed_precision")
        self._gpu_accelerator_add(self.config, _gpu_accelerator_trt(precision))

    def remove_trt(self):
        """Remove a TensorRT accelerator from the config."""
        if self._gpu_accelerator_status(self.config, "tensorrt"):
            self._gpu_accelerator_delete(self.config, "tensorrt")
