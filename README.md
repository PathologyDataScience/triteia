# triton_testing

### Repository organization

- /benchmarking - scripts and notebooks for generating official benchmarking results
- /configurations - contains .pbtxt files defining triton model serving configurations
- /notebooks - development notebooks for testing and demonstrating concepts
- /results - contains outputs of benchmarking experiments
- /simple_triton - contains repository code for simple_triton tools

### Using the Docker Engine Utility for Running A Container

- As a user, run the container interactively.
 docker run \
  --gpus=1 \
  --ipc=host --rm \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --net=host \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/mar9654/tritonClient/models:/models \
  nvcr.io/nvidia/tritonserver:22.07-py3 \
  tritonserver \
  --model-repository=/models \
  --exit-on-error=false \
  --model-control-mode=poll \
  --repository-poll-secs 30 ![image](https://user-images.githubusercontent.com/30137669/189497450-6ab6d43f-155b-4a20-b5d3-050f5adc490c.png)

Client libraries are installed on Docker running Jupyter notebook and run seprately.
For this experiiment both server and client are running on the same machine. 
