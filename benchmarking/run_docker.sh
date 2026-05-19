#!/usr/bin/env bash
#
# This file launches docker and mounts some directories
# if you have issues with missing files in the server: 
# The model directory will be also be mounted in the server, which is launched from THIS CLIENT FILE
# in other words, a recursive mount: docker run -v /dir:/dir .... docker run -v /dir:/dir
# keep in mind that recursive mounts with docker will always rely on  HOST paths, not from within a container

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
  -v "$PWD/models/":/home/$USER/triteia/models/ \
  -v /home/aza4423/BENCHMARK_DATA/:/data:ro \
  -v "$PWD/test_data":/home/$USER/triteia/test_data/ \
  -v /data/anders_aza4423/triteia_results/:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --rm \
  --name triteia_test_$USER \
  -it triteia:benchmark
