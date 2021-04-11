# https://hub.docker.com/r/cschranz/gpu-jupyter
# This is a Dockerfile which has done all the heavy lifting to ensure
#    that there is a container which can run DL on GPU. Running this
#    and using GPU still requires that you do some prep work to ensure
#    that the underlying operating system has some nvidia utilities.
# Long term if this project takes off it could winnow down to a minimal
#    docker environment; this one is quite heavy. It *is sufficient* for
#    testing and building, though.
FROM cschranz/gpu-jupyter:v1.4_cuda-11.0_ubuntu-18.04_slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# copy project requirement files here to ensure they will be cached.
COPY pyproject.toml /install/pyproject.toml
COPY src /install/src

WORKDIR /install
# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN pip install .

WORKDIR /home/jovyan
# Need to copy from builder into base
# Using https://github.com/python-poetry/poetry/discussions/1879 as a resource
