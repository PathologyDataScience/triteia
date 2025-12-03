import json

import torch
import triton_python_backend_utils as pb_utils
from torch.nn import Module
from torch.utils.dlpack import to_dlpack

class EmptyModule(Module):
    def __init__(self):
        super().__init__()
    def forward(self, _):
        return torch.tensor([1])

class TritonPythonModel:
    def initialize(self, args):
        """
        This will create an empty model that will read an input and always return the value "1"
        """

        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        self.model = torch.nn.Sequential(EmptyModule()) \
            .to(self.device) \
            .eval()

        self.gpu_id = args.get("model_instance_device_id", 0)
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

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
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_0")
            input_np = input_tensor.as_numpy()

            with torch.no_grad():
                result = self.model(torch.tensor(input_np, device=self.device))
            out_tensor = pb_utils.Tensor.from_dlpack("output_0", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
