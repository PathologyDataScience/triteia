#!/usr/bin/env bash

wsi_path="${1:?Error: WSI path must be provided as \$1}"
modelname="${2:?Error: Model name must be provided as \$2}"
output_dir="${3:?Error: Output directory must be provided as \$3}"
CLEAR_CACHE_REMOTELY="${4:-true}"
gpus="${5:-1,2,4,6,8}"
concurrencies="${6:-10}"

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

cmd="python ./benchmarking/triton_benchmark.py --numpy --inference-only --output ${output_dir} --model-name ${modelname} --wsi-path ${wsi_path} --batch-size 128 --url localhost:7985 --metrics-endpoint localhost:7986/metrics"

if [[ "${modelname}" == "resnet50" || "${modelname}" == "resnet50_trt_uint8" || "${modelname}" == "gigapath_trt_uint8" ]]; then
  cmd="${cmd} --nchw"
fi

# Parse comma-separated GPU list
IFS=',' read -ra gpu_array <<< "$gpus"

# Parse comma-separated batch size list
IFS=',' read -ra concurrency_array <<< "$concurrencies"

for gpu in "${gpu_array[@]}"
do
  zgpu=$((gpu - 1))
  gpu_string="$(seq 0 1 $zgpu | tr '\n' ' ')"

  for concurrency in "${concurrency_array[@]}"
  do
    tb_name="triton${modelname}multiusergpu${gpu}c${concurrency}n1"
    test -d "${output_dir}/$tb_name" && continue
    set -xe
    if [ ! "$(docker ps -q -f name=tritonserver_$USER)" ]; then
        ./launch_server.sh --num-gpus $gpu --start-gpu-id 0 ${modelname} --detached 1 --http-port 7984 --grpc-port 7985 --metrics-port 7986
        sleep 120
    fi
    clear_cache
    set +xe

    # Ensure that there is always at least 2x$gpu connections to the GPUs
    if [[ $gpu -gt 4 && $concurrency -lt 4 ]]; then
      limit=$gpu
    else
      limit=4
    fi
    
    for num_clients in $(seq 1 1 "${concurrency}")
    do
      tb_name="triton${modelname}multiusergpu${gpu}c${concurrency}n${num_clients}"
      set -xe
      eval "$cmd --limit $limit --tensorboard-name ${tb_name}" &
      set +xe
    done

    echo "Waiting for background jobs .."
    wait # won't continue until all above jobs are finished
    echo "Done"
  done

  if [ "$(docker ps -q -f name=tritonserver_$USER)" ]; then
    docker container stop tritonserver_$USER
    sleep 10 # seems to be necessary - docker container stop does not properly clean up right away
  fi
done
