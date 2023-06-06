#!/bin/bash
#!/bin/sh
filename="/tf/notebooks/benchmarking_results/warmup_results/ConvNeXtXLarge/ConvNeXtXLarge_amp_Batch64_GPU8_iter5_BatchTest.txt"
echo $filename
echo -e "Benchmark using Model: ConvNeXtXLarge, Batch Test, Optimization: amp, Batch size: (32,64,96,128), GPU size: (8), iterations: 5\n" 
echo -e "Benchmark using Model: ConvNeXtXLarge, Batch Test, Optimization: amp, Batch size: (32,64,96,128), GPU size: (8), iterations: 5\n" >> $filename
gpu_num=6
while [ $gpu_num -ne 8 ]
        do
                gpu_num=$(($gpu_num+2))
                limit=8
                while [ $limit -ne 10 ]
                        do
                                if [ $limit -ge 128 ]; then
                                        limit=$(($limit+128))
                                elif [ $limit -ge 64 ]; then
                                        limit=$(($limit+64))
                                else
                                        limit=$(($limit+2))
                                fi
                                maxbatchsize=32
                                while [ $maxbatchsize -ne 128 ]
                                        do
                                                maxbatchsize=$(($maxbatchsize+32))
                                                echo "gpu_num:$gpu_num  Workers: $limit maxbatchsize: $maxbatchsize"
                                                python /tf/notebooks/simple_triton/benchmarking/benchmark_interface.py  --limit 1 --gpu-num $gpu_num  --fileoutput $filename   --iterations 5 --use-amp --precision "FP16"  --maxbatchsize $maxbatchsize  --model-name "ConvNeXtXLarge"
                                                echo "==============================================================="
                                        done

                        done
                        echo -e "" >> $filename
        done

# "convnextsmall.tensorflow"
#"ConvNeXtXLarge"
# --gpu-count $limit
# --workers
# --use-trt --precision "FP16"
# --maxbatchsize $limit
# --model-name  "convnext-tiny"
