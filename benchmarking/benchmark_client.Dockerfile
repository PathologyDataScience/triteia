# for the paper, this docker image is built with:
# docker build -f client.Dockerfile . -t triteia:benchmark --build-arg DOCKER_GROUP_ID=$(getent group docker | cut -d: -f3) --build-arg UID=$(id -u) --build-arg GID=$(id -g)  --build-arg USERNAME=$USER
FROM python:3.10-slim AS build-image
ARG USERNAME=myuser

# create and activate virtual environment
RUN python3 -m venv /home/$USERNAME/venv
ENV PATH="/home/$USERNAME/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && apt install -y rdfind gcc git libc6-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install triteia
WORKDIR /home/$USERNAME/code/triteia
COPY triteia/ triteia
COPY pyproject.toml .
# comment out scm (i.e. git) line in pyproject.toml
RUN sed -i 's/.*\[tool.setuptools_scm\]/#&/g' pyproject.toml

RUN pip3 install --no-cache-dir .
# de-duplicate files and replace them with symlinks
RUN rdfind -minsize 32768 -makehardlinks true -makeresultsfile false /home/$USERNAME

FROM superlinear/python-gpu:3.10-cuda11.8 AS run-image

ARG UID=1009
ARG GID=3000
ARG USERNAME=myuser
RUN addgroup --gid $GID $USERNAME && adduser --uid $UID --ingroup $USERNAME $USERNAME
WORKDIR /home/$USERNAME

COPY --from=build-image --chown=$USERNAME:$USERNAME /home/$USERNAME/venv /home/$USERNAME/venv
 
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && \
    apt install -y libtiff-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

USER $USERNAME
WORKDIR /home/$USERNAME/triteia
COPY --chown=$USERNAME:$USERNAME triteia/ triteia
COPY --chown=$USERNAME:$USERNAME pyproject.toml .


# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/$USERNAME/venv
ENV PATH="/home/$USERNAME/venv/bin:$PATH"

CMD ["/usr/bin/env", "bash"]

# optional install of docker engine for test mode so that triton server container can be launched from the client container
# this is useful for testing, which will start and restart servers several times for different use cases
# we also install jupyter-lab for notebooks
FROM run-image AS test
USER root
ARG DOCKER_GROUP_ID
ARG USERNAME=myuser
RUN groupadd -g ${DOCKER_GROUP_ID} docker && usermod -aG ${DOCKER_GROUP_ID} $USERNAME
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
RUN chown $USERNAME:$USERNAME /home/$USERNAME/triteia/
USER $USERNAME

COPY --chown=$USERNAME:$USERNAME README.md pyproject.toml ./
# Install CPU-version of torch explicitly so that other dependencies doesn't default to pull the GPU version
# (for example, packages like `timm` or `lightly` depend on torch, and they might pull the GPU version when they are installed) 
RUN pip3 install --no-cache-dir torch torchvision timm transformers
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip3 install --no-cache-dir .[examples] tox pytest

COPY --chown=$USERNAME:$USERNAME tox.ini server.Dockerfile ./
COPY --chown=$USERNAME:$USERNAME tests tests
COPY --chown=$USERNAME:$USERNAME models models
