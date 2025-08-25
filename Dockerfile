from debian:stable-slim as base

env CELLPOSE_LOCAL_MODELS_PATH=/root/.cellpose/models

# python dep installation
from base as py-base

# set up uv/venv/workdir
copy --from=ghcr.io/astral-sh/uv:latest /uv /bin/
env UV_COMPILE_BYTECODE=1
run ["uv", "venv", "-p", "3.12", ".venv"]
run ["mkdir", "/workdir"]
WORKDIR /workdir

# install deps
copy ./pyproject.toml .
run ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"]
run ["uv", "pip", "install", "-r", "requirements.txt"]
run ["/.venv/bin/python", "-c", "from cellpose import models; models.CellposeModel()"]

# install package
copy ./src ./src 
run ["uv", "pip", "install", "."]

from base
copy --from=py-base /root/.local/share/uv /root/.local/share/uv
copy --from=py-base /.venv /.venv
env PATH=/.venv/bin:$PATH

copy --from=py-base /root/.cellpose/models /root/.cellpose/models
run ["chmod", "-R", "777", "/root/.cellpose/models"]
run ["umat", "-h"]