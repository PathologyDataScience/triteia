"""Triton Python backend model for hibou-L on Quest.

See https://huggingface.co/histai/hibou-L

IMPORTANT: this model loads with trust_remote_code=True, which means
transformers executes custom modeling code that normally comes FROM THE HUB
alongside the weights. Compute nodes have no network access, so those .py files
must be present in the staged directory - download the whole repository, not
just the checkpoint. The check in initialize() below reports this specifically,
because the error transformers raises otherwise is not obvious.

The preprocessing transform is hardcoded upstream (not loaded from the hub), so
it is unchanged here.
"""

import json
import os

import numpy as np
import torch
import torchvision
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils
from PIL import Image
from transformers import AutoModel


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

MODEL_NAME = "hibou-L"

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


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(model_config):
        """Returns a minimal model configuration for the hibou-L model.

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
        outputs = [{"name": "output_0", "data_type": "TYPE_FP32", "dims": [768]}]
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
        """Load hibou-L from weights staged in the user's weights directory."""

        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        weights_dir = _weights_dir()

        # trust_remote_code needs the repository's own modeling .py files. Say
        # so plainly rather than letting transformers fail while trying to
        # fetch them from a node with no network access.
        if not [f for f in os.listdir(weights_dir) if f.endswith(".py")]:
            raise RuntimeError(
                f"'{weights_dir}' contains no .py modeling files, but hibou-L "
                f"loads with trust_remote_code=True and needs the repository's "
                f"custom modeling code alongside the weights. Download the "
                f"whole repository, not just the checkpoint."
            )

        print(f"[triteia] hibou-L: loading from {weights_dir}", flush=True)

        # Preprocessing constants are hardcoded upstream rather than read from
        # the hub, so they are unchanged.
        self.transform = transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816]
                ),
            ]
        )
        self.model = AutoModel.from_pretrained(
            weights_dir, trust_remote_code=True, local_files_only=True
        )
        self.model = self.model.to(torch.device(f"cuda:{self.gpu_id}"))
        self.model.eval()
        print("[triteia] hibou-L: ready", flush=True)

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
                    features = feature_emb.pooler_output.detach().cpu().numpy()

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
