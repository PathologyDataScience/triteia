import argparse
import torch
import torch.nn as nn
import torchvision.models as models

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

parser = argparse.ArgumentParser(description="Generate and export a resnet50 model to ONNX")
parser.add_argument(
        "--uint8",
        action="store_true",
        default=False,
        help="Use a model with native uint8 instead of the default np.float32",
        )
args = parser.parse_args()

DTYPE_IS_UINT8 = args.uint8

class ModelUint8(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        # x is uint8
        x = x.to(torch.float32) / 255.0
        x = (x - self.mean) / self.std
        return self.m(x)

base = models.resnet50(pretrained=True).eval()
model = ModelUint8(base).eval() if DTYPE_IS_UINT8 else base

batch_size = 1 # can be any number, placeholder
x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.uint8 if DTYPE_IS_UINT8 else torch.float32)

# enable dynamic batch sizes
dynamic_axes = {'input_0': {0: 'batch'}, 'output_0': {0: 'batch'}}

filename="resnet50_" + ("uint8" if DTYPE_IS_UINT8 else "fp32") + ".onnx"
torch.onnx.export(model,
                  (x,),
                  filename,
                  export_params=True,
                  dynamic_axes=dynamic_axes,
                  input_names = ['input_0'],
                  output_names = ['output_0'],
                  )
