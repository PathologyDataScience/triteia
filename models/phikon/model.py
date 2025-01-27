import json
import numpy as np
import torch
from transformers import AutoImageProcessor, ViTModel
import triton_python_backend_utils as pb_utils
import tritonclient.utils as triton_utils


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
        model_config.set_max_batch_size(128)
        model_config.set_dynamic_batching()

        return model_config

    def initialize(self, args):
        """Initialize the phikon model from https://huggingface.co/owkin/phikon"""
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.model = self.model.to(torch.device(f"cuda:{self.gpu_id}"))
        self.image_processor = AutoImageProcessor.from_pretrained(
            "owkin/phikon", use_fast=True
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
                inputs = self.image_processor(input_np, return_tensors="pt").to(
                    torch.device(f"cuda:{self.gpu_id}")
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
