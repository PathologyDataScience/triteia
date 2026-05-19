#!/usr/bin/env bash

# cd into the same directory as this file is located
cd "$(dirname "$0")"
git_root_dir="$(git rev-parse --show-toplevel)"

# This version should match triton's docker image version
nvidia_version="${1:-25.02-py3}"

set -xe 


docker run -it --gpus all -w /resnet50_eg -v $PWD:/resnet50_eg nvcr.io/nvidia/pytorch:${nvidia_version} python export_resnet_to_onnx.py --uint8
docker run -it --gpus all -w /trt_optimize -v $PWD:/trt_optimize nvcr.io/nvidia/tensorrt:${nvidia_version} trtexec --onnx=resnet50_uint8.onnx --saveEngine=modeluint8.plan --useCudaGraph  --minShapes=input_0:1x3x224x224 --optShapes=input_0:16x3x224x224 --maxShapes=input_0:256x3x224x224

set +xe

model_dst_dir="${git_root_dir}/models/resnet50_trt_uint8"
test -d "${model_dst_dir}/1" || mkdir -p "${model_dst_dir}/1"
dst="${model_dst_dir}/config.pbtxt"
cp -v config.pbtxt "${dst}"
# we need to replace TYPE_FP32 with UINT8. But not for the model outputs.
# We could write an advanced algorithm to only modify two out of three lines,
# but instead, we've just added an exess space in the line "data_type:  TYPE_FP32" so that the output data_type remains unchanged.
sed -i 's/data_type: TYPE_FP32/data_type: TYPE_UINT8/g ; s/trt_float32/trt_uint8/g' ${dst}
cp -v modeluint8.plan "${model_dst_dir}/1/model.plan"

# ####
# FP32
# ####

set -xe
docker run -it --gpus all -w /resnet50_eg -v $PWD:/resnet50_eg nvcr.io/nvidia/pytorch:${nvidia_version} python export_resnet_to_onnx.py
docker run -it --gpus all -w /trt_optimize -v $PWD:/trt_optimize nvcr.io/nvidia/tensorrt:${nvidia_version} trtexec --onnx=resnet50_fp32.onnx --saveEngine=modelfp32.plan --useCudaGraph  --minShapes=input_0:1x3x224x224 --optShapes=input_0:16x3x224x224 --maxShapes=input_0:256x3x224x224
set +xe
model_dst_dir="${git_root_dir}/models/resnet50_trt_float32"
test -d "${model_dst_dir}/1" || mkdir -p "${model_dst_dir}/1"
dst="${model_dst_dir}/config.pbtxt"
cp config.pbtxt "${dst}"
cp -v modelfp32.plan "${model_dst_dir}/1/model.plan"

ls -R "${git_root_dir}/models/resnet50_trt_"*
