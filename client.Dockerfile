FROM python:3.10-slim AS build-image

# create and activate virtual environment
RUN python3 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && apt install -y rdfind gcc git libc6-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install simple-triton
WORKDIR /home/myuser/code/simple_triton
COPY simple_triton/ simple_triton
COPY pyproject.toml .
# comment out scm (i.e. git) line in pyproject.toml
RUN sed -i 's/.*\[tool.setuptools_scm\]/#&/g' pyproject.toml

RUN pip3 install --no-cache-dir .
# de-duplicate files and replace them with symlinks
RUN rdfind -minsize 32768 -makehardlinks true -makeresultsfile false /home/myuser

FROM python:3.10-slim AS run-image

RUN useradd --create-home myuser
COPY --from=build-image --chown=myuser:myuser /home/myuser/venv /home/myuser/venv

ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && \
    apt install -y libtiff-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

USER myuser
WORKDIR /home/myuser/simple_triton
COPY --chown=myuser:myuser simple_triton/ simple_triton
COPY --chown=myuser:myuser pyproject.toml .


# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

CMD ["/usr/bin/env", "bash"]

# optional install of docker engine for test mode so that triton server container can be launched from the client container
# this is useful for testing, which will start and restart servers several times for different use cases
# we also install jupyter-lab for notebooks
FROM run-image AS test
USER root
ARG DOCKER_GROUP_ID
RUN groupadd -g ${DOCKER_GROUP_ID} docker && usermod -aG ${DOCKER_GROUP_ID} myuser
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && \
    apt install -y curl git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://get.docker.com | sh
# for jupyter notebooks as non-root
RUN mkdir --mode a+rxw /.local /.jupyter /.cache /models/ /.config
RUN chown myuser:myuser /home/myuser/simple_triton/
USER myuser

# comment out scm (i.e. git) line in pyproject.toml
RUN sed -i 's/.*\[tool.setuptools_scm\]/#&/g' pyproject.toml
COPY --chown=myuser:myuser README.md pyproject.toml ./
# Install CPU-version of torch explicitly so that other dependencies doesn't default to pull the GPU version
# (for example, packages like `timm` or `lightly` depend on torch, and they might pull the GPU version when they are installed) 
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir .[examples] tox pytest

COPY --chown=myuser:myuser tox.ini server.Dockerfile models_entrypoint.sh setup_repository.py ./
COPY --chown=myuser:myuser tests tests
COPY --chown=myuser:myuser models models
