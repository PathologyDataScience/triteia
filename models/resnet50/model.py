# Link: https://github.com/triton-inference-server/python_backend/tree/main/examples/instance_kind

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json

import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
from transformers import AutoImageProcessor, ResNetModel, AutoModel


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
        """
        # Here we set up the device onto which our model will beloaded,
        # based on specified `model_instance_kind` and `model_instance_device_id`
        # fields.
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)

        self.model = ResNetModel.from_pretrained("microsoft/resnet-50") \
            .to(self.device) \
            .eval()

        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.gpu_id = args.get("model_instance_device_id", 0)
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
