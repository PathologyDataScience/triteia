#!/usr/bin/env bash
# See README.md

OUTPUT_DIR="${1:-/results/}"
CLEAR_CACHE_REMOTELY="${2:-true}"

# Function to clear cache if enabled
clear_cache() {
  if [[ "${CLEAR_CACHE_REMOTELY,,}" == "true" ]]; then
    echo "Clearing cache remotely..."
    curl localhost:7987/run
  else
    echo "Skipping remote cache clear (CLEAR_CACHE_REMOTELY=false)"
  fi
}

for bs in 256 128 64 32
do
  for workers in 16 32 64 
  do
    for chunk in 448 896 1120
    do
      for prefetch in 8 16 4
      do
        outfile="${OUTPUT_DIR}/read_speeds/wsi_read_speed_b${bs}_w${workers}_p${prefetch}_c${chunk}.csv"
        if [[ -e ${outfile} ]]
        then
          echo "skipping run for batch size $bs chunk $chunk workers $workers prefetch $prefetch - results already exists"
          continue
        fi
        set -x
        # drop caches
        clear_cache
        # run ...
        python ./benchmarking/tileiterator_benchmark.py --wsi-path /data/5/ --batch-size $bs --chunk-size $chunk --workers $workers --prefetch $prefetch --output "${OUTPUT_DIR}/read_speeds"
        set +x
      done
    done
  done
done

# to find the fastest average read-speed:
# for file in wsi_*read_*.csv; do echo -n "$file"$'\t'; sed 1d $file | awk -F',' '{ sum += $6; count++ } END { print sum/count }' $file ; done | sort -nk2

for modelname in resnet50 uni gigapath resnet50_trt_uint8 #gigapath_trt_uint8
do
  ./benchmarking/triton_batchsize_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" false
done

#just in case
docker container stop tritonserver_$USER || true
 
for modelname in resnet50 uni gigapath resnet50_trt_uint8 #gigapath_trt_uint8
do
  ./benchmarking/triton_batchsize_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" 1 #1,8 256
done
# just in case
docker container stop tritonserver_$USER || true
# 
for modelname in resnet50 uni gigapath
do
  ./benchmarking/pytorch_batchsize_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" false
done

for modelname in resnet50 uni gigapath
do
  ./benchmarking/pytorch_batchsize_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" 1 #1,8 256
done

for modelname in resnet50 uni gigapath resnet50_trt_uint8 #gigapath_trt_uint8
do
  ./benchmarking/triton_multiuser_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}"
done
# just in case
docker container stop tritonserver_$USER || true


# to test the impact of "--limit" parameter. Did not find anything interesting (general rule of thumb: limit should about 2x number of GPUs)
# for modelname in resnet50 uni gigapath
# do
#   ./benchmarking/triton_limit_benchmark.sh /data/5 $modelname "${OUTPUT_DIR}" "${CLEAR_CACHE_REMOTELY}" 1 8 16,20
# done
# just in case
# docker container stop tritonserver_$USER || true
#

./benchmarking/gpu_vs_cpu_comparison.sh
 
echo "All benchmarks done"
