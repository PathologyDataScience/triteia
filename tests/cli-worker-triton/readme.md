## CLI to interact with Triton through Worker to perform Superpixel segmentation using CUDA-SLIC

- Example command to run Trion Inference Server with Models Path:  

docker run -it  --gpus '"device=0,1,2,3,4,5,6,7"'   --rm  --shm-size=4g --ulimit memlock=-1  -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864  -v /var/run/docker.sock:/var/run/docker.sock  -v /home/mar9654/temp/python_backend/build/models:/models  -e PYTHONPATH=/home/mar9654/temp/python_backend/examples/cuda_slic:$PYTHONPATH --name mar9654_triton_server nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver   --model-repository=/models  --exit-on-error=false  --strict-model-config=false --model-control-mode=explicit --load-model=add_sub 

- model.py and config.pbtxt are part of the model and placed in the respective directory.

In the docker shell, Install these dependencies. <br />
docker exec -it mar9654_triton_server /bin/bash <br />
pip install scikit-image <br />
pip install imagecodecs <br />
pip install cuda-slic <br />
pip install histomicstk --find-links https://girder.github.io/large_image_wheels


