#!/usr/bin/env bash

# Print welcome message
echo "This script will launch a docker container to start tritonserver with a given model loaded"

# cd into the same directory as this file is located
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage: launch_server.sh [OPTIONS] [MODEL_NAME]

Starts a Docker container running Triton Inference Server with a selected model.

Positional:
  MODEL_NAME                 Name of model directory under the model repository (if omitted, you will be prompted)

Options:
  --num-gpus N               Number of GPUs to expose to the container (default: 1; 0 disables GPUs)
  --start-gpu-id ID          Starting GPU ID when using multiple GPUs (default: 0)
  --detached 0|1             Run container in detached mode (default: 0)
  --image, --docker-image    Docker image name:tag to run (default: model-tritonserver:latest)
  --http-port PORT           Host HTTP port to map to Triton (default: 8000)
  --grpc-port PORT           Host gRPC port to map to Triton (default: 8001)
  --metrics-port PORT        Host Metrics port to map to Triton (default: 8002)
  -m, --models-dir PATH      Host path to models directory to mount (default: "$PWD/models")
  -h, --help                 Show this help and exit
EOF
}

# Default values
NUM_GPUS=1
START_GPU_ID=0
MODEL=""
IMAGE_NAME="model-tritonserver:latest"
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002
MODELS_DIR="$PWD/models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --start-gpu-id)
      START_GPU_ID="$2"
      shift 2
      ;;
    --detached)
      DETACHED="$2"
      shift 2
      ;;
    --image|--docker-image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --http-port)
      HTTP_PORT="$2"
      shift 2
      ;;
    --grpc-port)
      GRPC_PORT="$2"
      shift 2
      ;;
    --metrics-port)
      METRICS_PORT="$2"
      shift 2
      ;;
    -m|--models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    *)
      MODEL="$1"
      shift
      ;;
  esac
done

# Build a selectable list of models from MODELS_DIR if MODEL not provided explicitly
if [[ -z "$MODEL" ]]; then
  if [[ -d "$MODELS_DIR" ]]; then
    mapfile -t MODEL_DIRS < <(find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort)
    if [[ ${#MODEL_DIRS[@]} -gt 0 ]]; then
      echo "Select which models:"
      for i in "${!MODEL_DIRS[@]}"; do
        echo "  [$i] ${MODEL_DIRS[$i]}"
      done
      read -rp "Select model index: " idx
      if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx >= 0 && idx < ${#MODEL_DIRS[@]} )); then
        MODEL="${MODEL_DIRS[$idx]}"
      else
        echo "Invalid selection"; exit 1
      fi
    else
      echo "No models found in $MODELS_DIR"; exit 1
    fi
  else
    echo "Models directory not found: $MODELS_DIR"; exit 1
  fi
fi

# Construct GPU device string if GPUs are requested and available
GPU_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
else
  AVAILABLE_GPUS=0
fi

if [[ "$NUM_GPUS" -gt 0 && "$AVAILABLE_GPUS" -gt 0 ]]; then
  if [ "$NUM_GPUS" -eq 1 ]; then
    GPU_DEVICES="\"device=$START_GPU_ID\""
  else
    GPU_LIST=""
    for ((i=0; i<NUM_GPUS; i++)); do
      if [ $i -eq 0 ]; then
        GPU_LIST="$START_GPU_ID"
      else
        GPU_LIST="$GPU_LIST,$((START_GPU_ID + i))"
      fi
    done
    GPU_DEVICES="\"device=$GPU_LIST\""
  fi
  GPU_FLAG=(--gpus "$GPU_DEVICES")
else
  echo "No GPUs available or NUM_GPUS=0; running without --gpus"
fi

if [[ "$DETACHED" -eq 1 ]]; then
  DETACH_ARG="true"
else
  DETACH_ARG="false"
fi

if [[ -z "$HF_TOKEN" ]]
then
  echo "Warning: no HF_TOKEN defined, some models may be downloaded" > /dev/fd/2
  sleep 3 # give users a chance to ctrl+c
fi

set -xe

docker run \
  "${GPU_FLAG[@]}" \
  --rm \
  --detach="$DETACH_ARG" \
  -e HF_TOKEN="${HF_TOKEN}" \
  --name tritonserver_$USER \
  -p "${HTTP_PORT}:8000" -p "${GRPC_PORT}:8001" -p "${METRICS_PORT}:8002" -p 8003:8003 \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ipc=host \
  -v "$MODELS_DIR":/models \
  "$IMAGE_NAME" \
  tritonserver --model-repository=/models --model-control-mode=explicit --exit-on-error=false \
  --backend-config=default-max-batch-size=256 --metrics-config summary_latencies=true \
  --load-model "$MODEL"
