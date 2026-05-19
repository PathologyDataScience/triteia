#!/usr/bin/env bash
# See README.md
#
# This file creates number from CPU-only based preprocessing
# the model definitions are not included in the repository.
# to create them, duplicate the gigapath, uni and python into "_old" folders.  Change the device from "self.device" to CPU where appropriate.

OUTPUT_DIR="${1:-/results/}"
CLEAR_CACHE_REMOTELY="${2:-false}"

for modelname in uni_old.python gigapath_old.python resnet_old.python
do
  ./benchmarking/triton_batchsize_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" false
done

# for modelname in uni_old.python gigapath_old.python resnet_old.python
# do
#   ./benchmarking/triton_batchsize_benchmark.sh /data/7 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" 1
# done
#just in case
#docker container stop tritonserver_$USER || true
 
echo "All benchmarks done"
