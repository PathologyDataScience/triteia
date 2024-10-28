FROM nvcr.io/nvidia/tritonserver:24.07-py3
RUN pip install --no-cache-dir --upgrade pip

# common dependencies
RUN pip install --no-cache-dir tritonclient

# phikon
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir pillow

# uni
RUN pip install --no-cache-dir timm
RUN pip install --no-cache-dir huggingface-hub

# conch
RUN pip install --no-cache-dir git+https://github.com/Mahmoodlab/CONCH.git

COPY setup_repository.py /
COPY models_entrypoint.sh .
RUN chmod +x models_entrypoint.sh
ENTRYPOINT ["./models_entrypoint.sh"]
