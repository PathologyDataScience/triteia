# TensorRT to Triton
*based on the [quickstart guide from NVIDIA](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton)*

The script `export_model.sh` will download and export a pretrained ResNet50-model to .onnx, and then to a tensorRT compiled version.
It exports both a uint8 model and a FP32 model. The uint8 model is the same as the FP32 model, except that the conversion from uint8 to FP32 will happen on the server-side instead of client-side. With gRPC, this is much faster.

Both models will automatically be copied to this repositories' `model` folder. You can then launch the server with `./launch_server resnet50_trt_uint8`. This model is _much_ faster than regular PyTorch.