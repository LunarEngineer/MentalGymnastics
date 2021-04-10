# https://hub.docker.com/r/cschranz/gpu-jupyter
# This is a Dockerfile which has done all the heavy lifting to ensure
#    that there is a container which can run DL on GPU. Running this
#    and using GPU still requires that you do some prep work to ensure
#    that the underlying operating system has some nvidia utilities.
# Long term if this project takes off it could winnow down to a minimal
#    docker environment; this one is quite heavy. It *is sufficient* for
#    testing and building, though.
FROM python:3.8.6-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    POETRY_VERSION=1.1.5

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# Install basic dependencies for poetry install
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential \
        gcc \
        # This is in here for development purposes only.
        # When this goes live it needs to be removed (most likely)
        git

RUN pip install --upgrade setuptools pip
# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# copy project requirement files here to ensure they will be cached.
COPY pyproject.toml /install/pyproject.toml
COPY src /install/src

WORKDIR /install
# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --no-dev --no-root

FROM cschranz/gpu-jupyter:v1.3_cuda-10.2_ubuntu-18.04_slim as base

# What else needs to be done to set up an environment?
COPY --from=builder /install/.venv /usr/local/.mentalgym_env
WORKDIR /usr/local/.mentalgym_env
RUN source bin/activate
RUN python -m ipykernel install --user --name mentalgym --display-name "Mental Gym (Python 3.8.6)"
WORKDIR /home/jovyan
# Need to copy from builder into base
# Using https://github.com/python-poetry/poetry/discussions/1879 as a resource