"""Triton Python backend model for phikon on Quest.

Loads weights from local disk only. Quest compute nodes have no outbound
network access, so weights are pre-staged by stage_weights.sh from a login
node.

Weights are supplied by the USER via TRITEIA_WEIGHTS_DIR, a colon-separated
search path. Several models are licence-restricted and their owners require
per-user access, so this installation ships no weights.

Expects: ${TRITEIA_WEIGHTS_DIR}/phikon/ - a standard transformers model
directory. from_pretrained reads the model weights and the preprocessing
config from the same path, so there is no risk of mismatched normalisation
constants as there is with the timm-based models.
"""

import json
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils
import transformers
from transformers import AutoImageProcessor, ViTModel

MODEL_NAME = "phikon"

EMBED_DIM = 768


def _weights_dir():
    """Locate this model's weights in the user's weights directory.

    TRITEIA_WEIGHTS_DIR is supplied by the USER: several models are
    licence-restricted and their owners require per-user access, so each user
    downloads the models they are licensed for. Nothing is downloaded here -
    compute nodes have no outbound network access.

    Errors are deliberately specific: this runs inside the Triton Python
    backend, where a bare FileNotFoundError would surface to the user as an
    opaque model-load failure.
    """
    root = os.getenv("TRITEIA_WEIGHTS_DIR", "")
    if not root:
        raise RuntimeError(
            "TRITEIA_WEIGHTS_DIR is not set. Model weights are supplied by the "
            "user; point it at the directory holding your downloaded models."
        )
    if not os.path.isdir(root):
        raise RuntimeError(
            f"TRITEIA_WEIGHTS_DIR is '{root}', which is not a directory."
        )

    path = os.path.join(root, MODEL_NAME)
    if not os.path.isdir(path):
        try:
            present = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )
        except OSError:
            present = []
        raise RuntimeError(
            f"No weights for '{MODEL_NAME}': expected '{path}'. "
            f"{root} contains: {present if present else 'nothing'}."
        )

    contents = sorted(os.listdir(path))
    if not contents:
        raise RuntimeError(f"'{path}' is empty - the download never ran.")

    if not any(f.endswith((".safetensors", ".bin", ".pt", ".pth"))
               for f in contents):
        raise RuntimeError(
            f"'{path}' contains no checkpoint file (looked for .safetensors, "
            f".bin, .pt, .pth). It holds: {contents}. The HuggingFace client "
            f"creates the directory before fetching, so this usually means the "
            f"download failed or was refused."
        )

    return path


def _load_tolerant(loader, path, required, optional, label):
    """Call `loader(path, **required, **optional)`, dropping optional kwargs
    that this version of transformers rejects.

    Loader kwargs have shifted between transformers 4.x (which Lee's code was
    written against) and the 5.x in the server image. Each optional kwarg is
    dropped individually on TypeError so that a removal degrades gracefully
    instead of failing model load, and the outcome is logged.
    """
    remaining = dict(optional)
    dropped = []

    while True:
        try:
            obj = loader(path, **required, **remaining)
            if dropped:
                print(
                    f"[triteia] {label}: loaded without unsupported kwarg(s) "
                    f"{dropped} on transformers {transformers.__version__}",
                    flush=True,
                )
            else:
                print(f"[triteia] {label}: loaded with {list(remaining)}", flush=True)
            return obj
        except TypeError as exc:
            culprit = next(
                (k for k in remaining if k in str(exc)), None
            )
            if culprit is None:
                # TypeError unrelated to an optional kwarg - do not mask it.
                raise
            remaining.pop(culprit)
            dropped.append(culprit)


def _load_first_supported(loader, path, required, alternatives, label):
    """Call `loader` with the first kwarg set this version accepts.

    Used where a kwarg was renamed rather than removed, so dropping it is not
    equivalent - a different spelling must be tried instead. Alternatives are
    attempted in order; the last entry should be an empty dict so that the
    call still succeeds on versions accepting none of them.
    """
    last_exc = None
    for kwargs in alternatives:
        try:
            obj = loader(path, **required, **kwargs)
            print(
                f"[triteia] {label}: loaded with {kwargs or 'no optional kwargs'} "
                f"on transformers {transformers.__version__}",
                flush=True,
            )
            return obj
        except (TypeError, ValueError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"{label}: none of the attempted kwarg sets were accepted by "
        f"transformers {transformers.__version__}. Last error: {last_exc}"
    )


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(model_config):
        """Returns a minimal model configuration for the phikon model.

        Parameters
        ----------
        model_config : pb_utils.ModelConfig
          An object containing the existing model configuration.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [
            {
                "name": "input_0",
                "data_type": "TYPE_UINT8",
                "dims": [224, 224, 3],
            }
        ]
        outputs = [{"name": "output_0", "data_type": "TYPE_FP32", "dims": [EMBED_DIM]}]
        config = model_config.as_dict()
        input_names = [i["name"] for i in config["input"]]
        output_names = [i["name"] for i in config["output"]]
        for i in inputs:
            if i["name"] not in input_names:
                model_config.add_input(i)
        for o in outputs:
            if o["name"] not in output_names:
                model_config.add_output(o)
        model_config.set_max_batch_size(256)

        return model_config

    def initialize(self, args):
        """Load the pre-staged phikon model and image processor from disk."""
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.device = torch.device(f"cuda:{self.gpu_id}")

        weights_dir = _weights_dir()
        print(
            f"[triteia] {MODEL_NAME}: transformers {transformers.__version__}, "
            f"loading from {weights_dir}",
            flush=True,
        )

        # local_files_only makes a stray network attempt fail immediately with a
        # clear message instead of hanging on a compute node with no route out.
        #
        # add_pooling_layer / use_fast were written against transformers 4.x.
        # The SIF ships 5.x, where some loader kwargs were removed. Rather than
        # hardcode an assumption about which survived, attempt the call with the
        # kwarg and retry without it on TypeError. The Triton log records which
        # path was taken.
        self.model = _load_tolerant(
            ViTModel.from_pretrained,
            weights_dir,
            required={"local_files_only": True},
            optional={"add_pooling_layer": False},
            label=f"{MODEL_NAME} ViTModel",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # add_pooling_layer=False drops the pooler head. If the kwarg was not
        # accepted, the pooler exists but is unused - execute() reads the CLS
        # token out of last_hidden_state directly, so the output is unaffected
        # either way. Recorded so the difference is visible in the log.
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            print(
                f"[triteia] {MODEL_NAME}: note - pooler present but unused "
                f"(CLS token is read from last_hidden_state)",
                flush=True,
            )

        # transformers 5.x deprecates use_fast in favour of an explicit
        # backend. Request torchvision first (the successor to use_fast=True),
        # falling back to use_fast for older transformers.
        #
        # This is not cosmetic: the torchvision and PIL backends use different
        # resampling implementations, so they produce slightly different pixel
        # values and therefore slightly different embeddings. Whichever backend
        # is used here must match the one used for any reference embeddings
        # being compared against.
        self.image_processor = _load_first_supported(
            AutoImageProcessor.from_pretrained,
            weights_dir,
            required={"local_files_only": True},
            alternatives=[{"backend": "torchvision"}, {"use_fast": True}, {}],
            label=f"{MODEL_NAME} image processor",
        )
        print(
            f"[triteia] {MODEL_NAME}: image processor "
            f"{type(self.image_processor).__name__}",
            flush=True,
        )

        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is not None and hidden != EMBED_DIM:
            raise RuntimeError(
                f"{MODEL_NAME}: config hidden_size {hidden} does not match the "
                f"declared output dim {EMBED_DIM}. auto_complete_config would "
                f"advertise the wrong tensor shape to clients."
            )

        print(f"[triteia] {MODEL_NAME}: ready (embed dim {EMBED_DIM})", flush=True)

    def execute(self, requests):
        """This function receives the requests (tiles) 'pb_utils.InferenceRequest'
        and performs the inference and returns the features for further processing.
        """
        responses = []

        for request in requests:

            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
                input_np = in_0.as_numpy()
                inputs = self.image_processor(input_np, return_tensors="pt").to(
                    self.device
                )

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.float16),
                ):
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]  # shape (1, 768)
                    features = features.detach().cpu().numpy()

                out_tensor_features = pb_utils.Tensor(
                    "output_0", features.astype(np.float32)
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_features]
                )
                responses.append(inference_response)
            except triton_utils.InferenceServerException as e:
                print("An error occured in Inference")
                print(e)

        return responses
