FROM nvcr.io/nvidia/tritonserver:24.12-py3

# common dependencies
RUN pip install --no-cache-dir tritonclient

# phikon
RUN pip install --no-cache-dir torch transformers pillow
# uni
RUN pip install --no-cache-dir timm huggingface-hub

# conch
RUN pip install --no-cache-dir git+https://github.com/Mahmoodlab/CONCH.git

# prevent creation of .pyc files owned by root
RUN mkdir /pycache
ENV PYTHONPYCACHEPREFIX=/pycache
