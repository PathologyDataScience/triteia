FROM python:3.10-slim as builder-image

# create and activate virtual environment
RUN python3 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && apt install -y gcc libc6-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install requirements
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir 'large_image[tiff]' histomics_stream -r requirements.txt

# install simple-triton
WORKDIR /home/myuser/code/simple_triton
COPY simple_triton/ simple_triton
COPY pyproject.toml .
# comment out scm line in pyproject.toml so we do not need to rebuild docker image when we change non-python files
# the alternative to commenting out is to either copy .git from host (but then we get personal files in the image)
# or to RUN git init
RUN sed -i 's/.*\[tool.setuptools_scm\]/#&/g' pyproject.toml
RUN pip3 install --no-cache-dir .

FROM python:3.10-slim AS runner-image

RUN useradd --create-home myuser
COPY --from=builder-image --chown=myuser:myuser /home/myuser/venv /home/myuser/venv

WORKDIR /home/myuser/simple_triton
RUN chown -R myuser:myuser /home/myuser/simple_triton
COPY --chown=myuser:myuser . .

ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \
    apt update && \
    apt install -y libtiff-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

USER myuser

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

CMD ["/usr/bin/env", "bash"]
