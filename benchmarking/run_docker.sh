#!/usr/bin/env bash

# make sure we are in root directory of the github repo
cd "$(dirname "$0")"/..

echo "remember to run pip install pynvml"
set -xe

docker run \
  --gpus all \
  -e USER=$USER \
  -e DOCKER_API_VERSION=1.41 \
  --security-opt seccomp:unconfined \
  --privileged \
  --ulimit core=0 \
  --network=host \
  --init \
  --shm-size=20g \
  -v "$PWD/launch_server.sh":/home/$USER/simple_triton/launch_server.sh \
  -v "$PWD/benchmarking":/home/$USER/simple_triton/benchmarking \
  -v /home/aza4423/BENCHMARK_DATA/:/data:ro \
  -v "$PWD/test_data":/home/$USER/simple_triton/test_data/ \
  -v /data/anders_aza4423/simple_triton_results/:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --rm \
  --name tritonclient_test_$USER \
  -it simple_triton_client:benchmark