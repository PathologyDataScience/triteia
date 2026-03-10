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
  -v "$PWD/launch_server.sh":/home/$USER/triteia/launch_server.sh \
  -v "$PWD/benchmarking":/home/$USER/triteia/benchmarking \
  -v ../BENCHMARK_DATA/:/data:ro \
  -v "$PWD/test_data":/home/$USER/triteia/test_data/ \
  -v ./results/:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --rm \
  --name triteia_test_$USER \
  -it triteia:benchmark