from huggingface_hub import login
import json
import numpy as np
import os
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import torch
from torchvision import transforms
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(model_config):
        """Returns a minimal model configuration for the virchow2 model.

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
        model_config.set_max_batch_size(128)
        model_config.set_dynamic_batching()

        return model_config

    def initialize(self, args):
        """This function initializes the uni model from hugging face.
        Requires setting environment variable `HF_TOKEN` with read-access to the
        huggingface gigpath repository https://huggingface.co/paige-ai/Virchow2
        """

        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        login(
            os.getenv("HF_TOKEN")
        )  # User Access Token, found at https://huggingface.co/settings/tokens
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.eval()
        self.model = self.model.to(torch.device(f"cuda:{self.gpu_id}"))
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
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
                patch_tokens = feature_emb[:, 5:]
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
