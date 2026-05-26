import json

import numpy as np
import timm
import os
import torch
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils
from huggingface_hub import get_token

def require_hf_token(repo_id):

    token =  get_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face access token for "
            f"{repo_id}. Set HF_TOKEN to a token with read access to the repository."
        )

    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token


class TritonPythonModel:
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
                "dims": [224, 224, 3],
            }
        ]
        outputs = [{"name": "output_0", "data_type": "TYPE_FP32", "dims": [1024]}]
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
        """This function initializes the uni model from hugging face.
        Requires setting environment variable `HF_TOKEN` with read-access to the
        huggingface uni repository https://huggingface.co/MahmoodLab/UNI
        """

        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        require_hf_token("MahmoodLab/UNI")
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model = self.model.to(self.device)
        self.mean = (
            torch.tensor(self.model.pretrained_cfg["mean"])
            .view(1, 3, 1, 1)
            .to(self.device)
        )
        self.std = (
            torch.tensor(self.model.pretrained_cfg["std"])
            .view(1, 3, 1, 1)
            .to(self.device)
        )

        self.model.eval()

    def execute(self, requests):
        """This function receives the requests (tiles) 'pb_utils.InfrerenceRequest'
        and performs the inference and returns the features for further processing.
        """

        responses = []
        for request in requests:

            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
                in_t = torch.from_numpy(in_0.as_numpy()).to(self.device)
                in_t = (in_t / 255.0).permute(0, 3, 1, 2)
                in_t = (in_t - self.mean) / self.std

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.float16),
                ):
                    features = self.model(in_t).detach().cpu().numpy()

                out_tensor_features = pb_utils.Tensor(
                    "output_0", features.astype(np.float32)
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_features]
                )
                responses.append(inference_response)
            except triton_utils.InferenceServerException as e:
                print("An error occurred in inference")
                print(e)

        return responses
