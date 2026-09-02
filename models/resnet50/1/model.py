"""Triton Python backend model for ResNet-50 on Quest.

Adapted from the Triton python_backend instance_kind example:
https://github.com/triton-inference-server/python_backend/tree/main/examples/instance_kind

Note this model differs from the pathology encoders: its input is CHW
[3, 224, 224] rather than HWC, and its 1000-dim output is ImageNet classifier
width rather than a learned embedding.

See https://huggingface.co/microsoft/resnet-50
"""

import json
import os

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
from transformers import AutoImageProcessor, ResNetModel


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

MODEL_NAME = "resnet50"

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



def _load_first_supported(loader, path, required, alternatives, label):
    """Call `loader` with the first kwarg set this transformers version accepts.

    The server image ships transformers 5.x while this code was written against
    4.x, and some loader kwargs were renamed rather than removed - so dropping
    one is not equivalent to re-spelling it. Alternatives are tried in order;
    the last entry should be empty so the call still succeeds where none apply.
    """
    last = None
    for kwargs in alternatives:
        try:
            obj = loader(path, **required, **kwargs)
            print(
                f"[triteia] {label}: loaded with "
                f"{kwargs if kwargs else 'no optional kwargs'}",
                flush=True,
            )
            return obj
        except (TypeError, ValueError) as exc:
            last = exc
            continue
    raise RuntimeError(
        f"{label}: none of the attempted kwarg sets were accepted by this "
        f"transformers version. Last error: {last}"
    )


class TritonPythonModel:
    def initialize(self, args):
        """
        This function initializes pre-trained ResNet50 model,
        depending on the value specified by an `instance_group` parameter
        in `config.pbtxt`.

        Depending on what `instance_group` was specified in
        the config.pbtxt file (KIND_CPU or KIND_GPU), the model instance
        will be initialised on a cpu, a gpu, or both. If `instance_group` was
        not specified in the config file, then models will be loaded onto
        the default device of the framework.

        Weights are read from the user's weights directory rather than
        downloaded; see the note at the top of this file.
        """
        # Here we set up the device onto which our model will beloaded,
        # based on specified `model_instance_kind` and `model_instance_device_id`
        # fields.
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"

        weights_dir = _weights_dir()
        print(f"[triteia] resnet50: loading from {weights_dir}", flush=True)

        # local_files_only makes a stray network call fail immediately with a
        # clear message rather than hanging on a node with no route out.
        self.processor = _load_first_supported(
            AutoImageProcessor.from_pretrained,
            weights_dir,
            required={"local_files_only": True},
            alternatives=[{"backend": "torchvision"}, {"use_fast": True}, {}],
            label="resnet50 image processor",
        )

        self.model = ResNetModel.from_pretrained(weights_dir, local_files_only=True) \
            .to(self.device) \
            .eval()

        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        print("[triteia] resnet50: ready", flush=True)

    @staticmethod
    def auto_complete_config(model_config):
        """Returns a minimal model configuration for the uni model.

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
                "dims": [3, 224, 224],
            }
        ]
        outputs = [{"name": "output_0", "data_type": "TYPE_FP32", "dims": [1000]}]
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

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = [None] * len(requests)

        with torch.inference_mode():
            for i, request in enumerate(requests):
                inputs = self.processor(pb_utils.get_input_tensor_by_name(request, "input_0").as_numpy(),
                                        return_tensors="pt",
                                        device=self.device)
                result = self.model(**inputs).pooler_output.flatten(1, -1)
                out_tensor = pb_utils.Tensor.from_dlpack("output_0", to_dlpack(result))
                responses[i] = pb_utils.InferenceResponse([out_tensor])
        return responses
