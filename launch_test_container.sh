#!/usr/bin/env bash

# This script is used to launch a container for testing purposes.
# First, it copies over the model definitions to a temporary directory
# Second, it creates a container for the client, mounting the docker socket from the host and the model definitions
# It then enters an interactive environment where users can run `pytest` or `tox`
#
# Usage: ./launch_test_container.sh
if [ -z "${HF_TOKEN}" ]; then
    printf "Warning: no HF_TOKEN defined, foundational model testing will not work\n"
    printf "You can add it by quitting this script/container, and then `export HF_TOKEN=...` and re-start the script\n"
fi
HF_TOKEN=${HF_TOKEN:-undefined}
model_tmp_directory="$(mktemp -d -t simple_triton_test_XXXXX)"
cp -r models/* "${model_tmp_directory}"/ || exit 1
chmod -R 777 "${model_tmp_directory}"
docker run --security-opt seccomp:unconfined --network=host -v "${model_tmp_directory}:${model_tmp_directory}:rw" -e TRITON_TMP_DIR="${model_tmp_directory}/" -e HF_TOKEN=${HF_TOKEN} --shm-size=2g -v /var/run/docker.sock:/var/run/docker.sock --rm --name tritonclient_test -it simple_triton_client:latest
# Clean up the temporary directory
printf "Cleaning up temporary directory: %s\n" "${model_tmp_directory}"
rm -rf "${model_tmp_directory}" 2>/dev/null || (docker run -v "${model_tmp_directory}:/tmp/" bash:latest bash -c "rm -r /tmp/*" && rmdir "$model_tmp_directory")
