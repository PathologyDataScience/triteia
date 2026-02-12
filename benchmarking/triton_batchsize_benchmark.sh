#!/usr/bin/env bash

wsi_path="${1:?Error: WSI path must be provided as \$1}"
modelname="${2:?Error: Model name must be provided as \$2}"
output_dir="${3:?Error: Output directory must be provided as \$3}"
CLEAR_CACHE_REMOTELY="${4:-true}"
inference_only="${5:-false}"
gpus="${6:-1,2,4,6,8}"
batch_sizes="${7:-32,64,128,256}"

if [[ -z "${HF_TOKEN}" ]]
then
  echo "No 'HF_TOKEN' environment variable defined, please \`export HF_TOKEN=...\`"
  exit 1
fi

# Function to clear cache if enabled
clear_cache() {
  if [[ "${CLEAR_CACHE_REMOTELY,,}" == "true" ]]; then
    echo "Clearing cache remotely..."
    curl localhost:7987/run
  else
    echo "Skipping remote cache clear (CLEAR_CACHE_REMOTELY=false)"
  fi
}

cmd="python ./benchmarking/triton_benchmark.py --model-name ${modelname} --wsi-path ${wsi_path} --output ${output_dir} --url localhost:7985 --metrics-endpoint localhost:7986/metrics"

if [[ "${inference_only}" == "1" || "${inference_only}" == "True" ]]; then
  cmd="${cmd} --inference-only"
fi

if [[ "${modelname}" == "resnet50" || "${modelname}" == "resnet50_trt_uint8" || "${modelname}" == "gigapath_trt_uint8" ]]; then
  cmd="${cmd} --nchw"
fi

# Parse comma-separated GPU list
IFS=',' read -ra gpu_array <<< "$gpus"

# Parse comma-separated batch size list
IFS=',' read -ra bs_array <<< "$batch_sizes"

for gpu in "${gpu_array[@]}"
do
  gpu=$((gpu))  # Convert to integer (trim whitespace)
  zgpu=$((gpu - 1))
  gpu_string="$(seq 0 1 $zgpu | tr '\n' ' ')"

  for bs in "${bs_array[@]}"
  do
    bs=$((bs))  # Convert to integer (trim whitespace)
    tb_name="triton${modelname}gpu${gpu}bs${bs}"
    if [[ "${inference_only}" == "1" || "${inference_only}" == "True" ]]; then
      tb_name="${tb_name}inferenceonly"
    fi
    test -d "${output_dir}/${tb_name}" && continue
    set -xe
    if [ ! "$(docker ps -q -f name=tritonserver_$USER)" ]; then
        ./launch_server.sh --num-gpus $gpu --start-gpu-id 0 ${modelname} --detached 1 --http-port 7984 --grpc-port 7985 --metrics-port 7986
        sleep 120
    fi
    clear_cache
    eval "$cmd --batch-size ${bs} --tensorboard-name ${tb_name} --gpus ${gpu_string}"
    set +xe
  done

  if [ "$(docker ps -q -f name=tritonserver_$USER)" ]
  then
    docker container stop tritonserver_$USER
    sleep 10 # seems to be necessary - docker container stop does not properly clean up right away
  fi
done
