FROM nvcr.io/nvidia/tritonserver:24.12-py3

# common dependencies
RUN pip install --no-cache-dir tritonclient

# phikon
RUN pip install --no-cache-dir torch transformers pillow
# uni
RUN pip install --no-cache-dir timm huggingface-hub

# conch
RUN pip install --no-cache-dir git+https://github.com/Mahmoodlab/CONCH.git

COPY setup_repository.py /
COPY models_entrypoint.sh .
RUN chmod +x models_entrypoint.sh
ENTRYPOINT ["./models_entrypoint.sh"]
