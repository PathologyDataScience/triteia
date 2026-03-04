#!/usr/bin/env bash
# This script will schedule a benchmark with different batch sizes and number of GPUS using pytorch.

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

cmd="python ./benchmarking/pytorch_benchmark.py --output ${output_dir} --model-name ${modelname} --wsi-path ${wsi_path}"

if [[ "${inference_only}" == "1" || "${inference_only}" == "True" ]]; then
  cmd="${cmd} --inference-only"
fi

if [[ "${modelname}" == "resnet50" ]]; then
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
    tb_name="pytorch${modelname}gpu${gpu}bs${bs}"
    if [[ "${inference_only}" == "1" || "${inference_only}" == "True" ]]; then
      tb_name="${tb_name}inferenceonly"
    fi
    test -d "${output_dir}/${tb_name}" && continue

    set -xe
    clear_cache
    eval "$cmd --batch-size ${bs} --tensorboard-name ${tb_name} --gpus ${gpu_string}"
    set +xe
    sleep 5
  done
done
