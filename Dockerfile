# 24.04 won't work since python3.12 is default which doesn't work with histomics_stream (because of numpy version)
FROM ubuntu:22.04 as builder-image

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

# apt-get update will fail on old docker installations.
# It's better to just build on another computer with an up-to-date docker installation.
# you can work around it with `--allow-insecure` but this is potentially harmful.
# ... git is necessary for pip installing simple_triton later
RUN apt-get update && apt-get install -y python3 python3-dev python3-venv python3-pip python3-wheel python3-openslide build-essential git && \
	apt-get clean && rm -rf /var/lib/apt/lists/*
 
# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"
 
# install requirements
RUN python -m ensurepip --upgrade
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir wheel setuptools 'large_image[tiff]' scikit_image --find-links https://girder.github.io/large_image_wheels
RUN pip3 install torch torchvision torchaudio
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r ./requirements.txt
# install simple-triton
WORKDIR /home/myuser/code/simple_triton
COPY simple_triton/ simple_triton
COPY pyproject.toml .
# comment out scm line in pyproject.toml so we do not need to rebuild docker image when we change non-python files
# the alternative to commenting out is to either copy .git from host (but then we get personal files in the image)
# or to RUN git init
RUN sed -i 's/.*\[tool.setuptools_scm\]/#&/g' pyproject.toml
RUN pip3 install .

FROM ubuntu:22.04 AS runner-image
RUN apt-get update && \
  apt-get install -y python3 python3-venv && \
# text editors are not _necessary_, but make things so much easier to debug
  apt-get install -y nano vim && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home myuser
COPY --from=builder-image --chown=myuser:myuser /home/myuser/venv /home/myuser/venv

USER myuser
WORKDIR /home/myuser/simple_triton
COPY --chown=myuser:myuser . .

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

CMD ["/usr/bin/env", "bash"]
