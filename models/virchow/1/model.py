"""Triton Python backend model for Virchow on Quest.

See https://huggingface.co/paige-ai/Virchow

The embedding is 2560 = 2 x 1280: the class token concatenated with the mean of
the patch tokens. Token 0 is the class token; patch tokens start at index 1.
"""

import json
import os

import numpy as np
import timm
import torch
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


# ---------------------------------------------------------------------------
# QUEST PORT: weights are loaded from the user's local directory.
#
# Quest compute nodes have no outbound network access, and several models are
# licence-restricted such that each user must obtain access themselves. Users
# download the models they are licensed for and set:
#
#     export TRITEIA_WEIGHTS_DIR=/path/to/my/weights
#
# with one subdirectory per model. Nothing is downloaded at load time and
# HF_TOKEN is never needed here.
# ---------------------------------------------------------------------------

MODEL_NAME = "virchow"

CHECKPOINT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
)


def _weights_dir():
    """Locate this model's weights in the user's weights directory.

    Errors are deliberately specific: this runs inside the Triton Python
    backend, where an unhandled exception reaches the user as an opaque
    model-load failure.
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
                d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
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
    return path


def _checkpoint(weights_dir):
    """Find the checkpoint file. Raises with a listing if there is none."""
    for name in CHECKPOINT_NAMES:
        p = os.path.join(weights_dir, name)
        if os.path.isfile(p):
            return p
    # Any other checkpoint-looking file, in case the repo uses another name.
    for f in sorted(os.listdir(weights_dir)):
        if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
            return os.path.join(weights_dir, f)
    raise RuntimeError(
        f"No checkpoint file in '{weights_dir}'. It holds "
        f"{sorted(os.listdir(weights_dir))}. The HuggingFace client creates the "
        f"directory before fetching, so this usually means the download failed "
        f"or was refused."
    )


def _load_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)



def _timm_from_local(weights_dir, fallback_args=None, **overrides):
    """Rebuild a timm model from a downloaded HuggingFace repo directory.

    timm publishes `architecture` and `model_args` in the repo's config.json -
    exactly what `timm.create_model("hf-hub:...")` would have used. Reading them
    reproduces the hub build without a network call, and without hardcoding
    architecture parameters here that could drift from the published model.

    Returns (model, pretrained_cfg). The pretrained_cfg dict carries the
    normalisation constants; getting those wrong does not raise anywhere, it
    just produces plausible and incorrect embeddings, so a missing config is
    treated as fatal rather than defaulted.
    """
    cfg_path = os.path.join(weights_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)

    arch = cfg.get("architecture")
    model_args = dict(cfg.get("model_args") or {})

    if not arch:
        if not fallback_args:
            raise RuntimeError(
                f"'{cfg_path}' does not declare an 'architecture', so the model "
                f"cannot be rebuilt locally. Re-download the full repository "
                f"rather than only the checkpoint."
            )
        arch = fallback_args["architecture"]
        model_args = dict(fallback_args.get("model_args") or {})
        print(
            f"[triteia] {MODEL_NAME}: config.json has no architecture; using "
            f"the published parameters compiled into this file",
            flush=True,
        )

    if "num_classes" in cfg and "num_classes" not in model_args:
        model_args["num_classes"] = cfg["num_classes"]
    model_args.update(overrides)

    print(f"[triteia] {MODEL_NAME}: building {arch} with {model_args}", flush=True)
    model = timm.create_model(arch, pretrained=False, **model_args)

    checkpoint = _checkpoint(weights_dir)
    state = _load_state_dict(checkpoint)
    # strict=True: a key or shape mismatch raises here rather than producing a
    # partially initialised model that returns meaningless embeddings.
    model.load_state_dict(state, strict=True)
    print(
        f"[triteia] {MODEL_NAME}: loaded {len(state)} tensors from "
        f"{os.path.basename(checkpoint)} (strict=True)",
        flush=True,
    )

    pretrained_cfg = dict(cfg.get("pretrained_cfg") or {})
    if "mean" not in pretrained_cfg or "std" not in pretrained_cfg:
        raise RuntimeError(
            f"No normalisation constants (mean/std) in '{cfg_path}'. Building "
            f"the architecture locally leaves timm's generic defaults in place, "
            f"which are not guaranteed to match this model and would silently "
            f"produce incorrect embeddings. Re-download the full repository."
        )
    # Keep the attribute consistent for anything that reads it.
    try:
        model.pretrained_cfg.update(pretrained_cfg)
    except (AttributeError, TypeError):
        model.pretrained_cfg = pretrained_cfg
    print(
        f"[triteia] {MODEL_NAME}: mean={pretrained_cfg['mean']} "
        f"std={pretrained_cfg['std']}",
        flush=True,
    )
    return model, pretrained_cfg


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(model_config):
        """Returns a minimal model configuration for the Virchow model.

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
        outputs = [{"name": "output_0", "data_type": "TYPE_FP32", "dims": [2560]}]
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
        """Build Virchow from weights staged in the user's weights directory."""
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        weights_dir = _weights_dir()
        # mlp_layer and act_layer are passed explicitly upstream and are not
        # JSON-serialisable, so they are always supplied as overrides rather
        # than read from the repo config.
        self.model, pretrained_cfg = _timm_from_local(
            weights_dir, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

        self.model = self.model.eval()
        self.model = self.model.to(torch.device(f"cuda:{self.gpu_id}"))
        self.transform = create_transform(
            **resolve_data_config(pretrained_cfg, model=self.model)
        )

    def execute(self, requests):
        """This function receives the requests (tiles) 'pb_utils.InfrerenceRequest'
        and performs the inference and returns the features for further processing.
        """

        responses = []
        for request in requests:

            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
                input_np = in_0.as_numpy()

                batch_size = input_np.shape[0]
                pil_images = [Image.fromarray(input_np[i]) for i in range(batch_size)]
                transformed_images = torch.stack(
                    [self.transform(img) for img in pil_images]
                )
                input_norm = transformed_images.to(torch.device(f"cuda:{self.gpu_id}"))
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.float16),
                ):
                    feature_emb = self.model(input_norm)
                class_tokens = feature_emb[:, 0]
                patch_tokens = feature_emb[:, 1:]
                embedding = torch.cat([class_tokens, patch_tokens.mean(1)], dim=-1)
                features = embedding.detach().cpu().numpy()
                out_tensor_features = pb_utils.Tensor("output_0", features)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_features]
                )
                responses.append(inference_response)
            except triton_utils.InferenceServerException as e:
                print("An error occured in Inference")
                print(e)

        return responses
