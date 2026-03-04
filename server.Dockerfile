FROM nvcr.io/nvidia/tritonserver:25.02-py3
# later versions of tritonserver deprecate tensorflow

# common dependencies
RUN pip install --no-cache-dir tritonclient

# phikon
RUN pip install --no-cache-dir torch transformers pillow
# uni
RUN pip install --no-cache-dir timm huggingface-hub

# conch
RUN pip install --no-cache-dir git+https://github.com/Mahmoodlab/CONCH.git

# avoid creating root-owned .pyc files in user repository
RUN mkdir /pycache
ENV PYTHONPYCACHEPREFIX=/pycache
