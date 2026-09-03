"""A drop-in replacement for `pytriton.client.ModelClient` built on tritonclient.

WHY THIS EXISTS
---------------
`nvidia-pytriton` publishes wheels tagged manylinux_2_35 (it requires
glibc >= 2.35, as noted in feature_extraction.py's own comment about
`from_existing_client`). Quest runs glibc 2.28, so pip's highest acceptable
tag is manylinux_2_28 and no installable distribution exists:

    ERROR: Could not find a version that satisfies the requirement
           nvidia-pytriton (from versions: none)

There is no packaging-level fix - no sdist is published, and the package
bundles a compiled Triton server binary. This affects any RHEL/Rocky 8 host,
which covers a large share of academic HPC, so it is a portability issue
rather than a Quest quirk.

`tritonclient` is already a triteia dependency, is the official client for the
server being used, and is what pytriton's own client wraps. Swapping to it
removes the glibc floor and ~1 GB of bundled server binary from the client
environment.

USAGE
-----
Interface-compatible with the subset of ModelClient that triteia uses:

    with GrpcModelClient("grpc://" + url, model_name, lazy_init=False,
                         init_timeout_s=10.0) as client:
        outputs = client.infer_batch(batch)   # -> {output_name: ndarray}
"""

import time

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype


class GrpcModelClient:
    """Minimal ModelClient-compatible wrapper over tritonclient.grpc.

    Implements only what triteia uses: construction with a grpc:// url,
    the context-manager protocol (for ExitStack.enter_context), and
    `infer_batch`.

    Parameters
    ----------
    url : str
        Server address. Accepts "grpc://host:port" or bare "host:port"; the
        scheme is stripped, since tritonclient expects host:port.
    model_name : str
        Name of a model already loaded on the server.
    model_version : str
        Model version. Empty string means "latest", matching pytriton.
    lazy_init : bool
        When False, verify the server and model are ready at construction
        rather than on first inference.
    init_timeout_s : float
        Seconds to wait for the model to become ready when lazy_init is False.
    inference_timeout_s : float
        Per-request timeout, passed through to infer().
    """

    def __init__(
        self,
        url,
        model_name,
        model_version="",
        lazy_init=False,
        init_timeout_s=10.0,
        inference_timeout_s=None,
        **_ignored,
    ):
        self._url = self._strip_scheme(url)
        self._model_name = model_name
        self._model_version = model_version
        self._inference_timeout_s = inference_timeout_s

        self._client = grpcclient.InferenceServerClient(url=self._url)

        # Input/output names and dtypes are read from the server rather than
        # assumed, so this works unchanged across models with different tensor
        # names or embedding dimensions.
        self._input_specs = None
        self._output_names = None

        if not lazy_init:
            self._wait_for_model(init_timeout_s)
            self._load_metadata()

    @staticmethod
    def _strip_scheme(url):
        for scheme in ("grpc://", "http://", "https://"):
            if url.startswith(scheme):
                return url[len(scheme) :]
        return url

    def _wait_for_model(self, timeout_s):
        deadline = time.time() + timeout_s
        last_error = None
        while time.time() < deadline:
            try:
                if self._client.is_server_ready() and self._client.is_model_ready(
                    self._model_name, self._model_version
                ):
                    return
            except Exception as exc:  # noqa: BLE001 - retried until timeout
                last_error = exc
            time.sleep(0.25)

        raise TimeoutError(
            f"Model '{self._model_name}' not ready at {self._url} after "
            f"{timeout_s}s. The server must be running and the model loaded "
            f"before inference. Last error: {last_error}"
        )

    def _load_metadata(self):
        meta = self._client.get_model_metadata(self._model_name, self._model_version)
        # (name, triton_datatype) per input, in the server's declared order.
        self._input_specs = [(i.name, i.datatype) for i in meta.inputs]
        self._output_names = [o.name for o in meta.outputs]

    def infer_batch(self, *args, **kwargs):
        """Run inference on a batch.

        Accepts the same shapes of call that triteia makes:

          * a single ndarray            -> bound to the first declared input
          * a dict of {name: ndarray}   -> bound by name
          * keyword arguments           -> bound by name

        Returns
        -------
        dict
            {output_name: numpy.ndarray}, matching pytriton's return type.
            feature_extraction.worker_infer takes the first value.
        """
        if self._input_specs is None:
            self._load_metadata()

        # --- normalise the call into {input_name: ndarray} ------------------
        if kwargs and not args:
            named = dict(kwargs)
        elif len(args) == 1 and isinstance(args[0], dict):
            named = dict(args[0])
        elif len(args) == 1:
            named = {self._input_specs[0][0]: args[0]}
        elif len(args) > 1:
            if len(args) > len(self._input_specs):
                raise ValueError(
                    f"{len(args)} positional inputs given but model "
                    f"'{self._model_name}' declares {len(self._input_specs)}"
                )
            named = {spec[0]: arr for spec, arr in zip(self._input_specs, args)}
        else:
            raise ValueError("infer_batch() requires at least one input array")

        spec_by_name = dict(self._input_specs)
        missing = [n for n in spec_by_name if n not in named]
        if missing:
            raise ValueError(
                f"Model '{self._model_name}' expects input(s) {missing} which "
                f"were not provided. Given: {sorted(named)}"
            )

        # --- build the request ---------------------------------------------
        infer_inputs = []
        for name, arr in named.items():
            if name not in spec_by_name:
                raise ValueError(
                    f"Model '{self._model_name}' has no input named '{name}'. "
                    f"Declared inputs: {sorted(spec_by_name)}"
                )
            datatype = spec_by_name[name]

            arr = np.asarray(arr)
            expected = triton_to_np_dtype(datatype)
            if expected is not None and arr.dtype != expected:
                # Cast rather than fail: the iterator's dtype is chosen from the
                # model config upstream, so a mismatch here is a narrow edge
                # case, but silently sending the wrong dtype would corrupt
                # results.
                arr = arr.astype(expected)
            arr = np.ascontiguousarray(arr)

            infer_input = grpcclient.InferInput(name, arr.shape, datatype)
            infer_input.set_data_from_numpy(arr)
            infer_inputs.append(infer_input)

        requested = [
            grpcclient.InferRequestedOutput(name) for name in self._output_names
        ]

        infer_kwargs = {}
        if self._inference_timeout_s is not None:
            infer_kwargs["client_timeout"] = self._inference_timeout_s

        response = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=requested,
            **infer_kwargs,
        )

        return {name: response.as_numpy(name) for name in self._output_names}

    # --- ModelClient compatibility surface ---------------------------------

    def close(self):
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing must not mask a real error
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @property
    def model_name(self):
        return self._model_name

    @property
    def url(self):
        return self._url
