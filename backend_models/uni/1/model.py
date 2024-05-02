import triton_python_backend_utils as pb_utils
import json
import numpy as np
import tritonclient.utils as triton_utils
import timm
import torch
from huggingface_hub import login
from PIL import Image
import os


class TritonPythonModel:

    def initialize(self, args):
        """
        This function initializes pre-trained UNI model from hugging face.
        The initialization is depending on the configuration from 'config.pbtxt'
        https://huggingface.co/MahmoodLab/UNI
        """
        # Load Model configuration
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output_0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        login(
            os.getenv("TOKEN")
        )  # login with your User Access Token, found at https://huggingface.co/settings/tokens

        # Initialize the pretrained UNI Model
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.model = self.model.to(torch.device(f"cuda:{self.gpu_id}"))

        # These values are obtained from UNI hugging face example
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.model.eval()

    def execute(self, requests):
        """
        This function receives the requests (tiles) 'pb_utils.InfrerenceRequest'
        and performs the inference and returns the features for further processing.
        """

        responses = []
        for request in requests:

            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
                input_np = in_0.as_numpy()
                input_np = np.transpose(input_np, (0, 3, 1, 2))
                input_tensor = torch.tensor(input_np)
                input_norm = (
                    input_tensor / 255.0 - self.mean.reshape((1, 3, 1, 1))
                ) / (self.std.reshape((1, 3, 1, 1)))
                input_norm = input_norm.float().to(torch.device(f"cuda:{self.gpu_id}"))

                with torch.inference_mode():
                    feature_emb = self.model(input_norm)
                    features = feature_emb.detach().cpu().numpy()
                    print(features)

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

    def finalize(self):
        print("Cleaning up ..")
