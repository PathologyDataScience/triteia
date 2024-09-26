FROM nvcr.io/nvidia/tritonserver:23.03-py3
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
