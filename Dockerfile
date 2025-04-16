from debian:stable-slim as base

env CELLPOSE_LOCAL_MODELS_PATH=/root/.cellpose/models

# python dep installation
from base as py-base
copy --from=ghcr.io/astral-sh/uv:latest /uv /bin/
env UV_COMPILE_BYTECODE=1
run ["uv", "venv", "-p", "3.12", ".venv"]
copy ./requirements.txt requirements.txt
run ["uv", "pip", "install", "-r", "requirements.txt"]
run ["/.venv/bin/python", "-c", "from cellpose import models; models.Cellpose(model_type=\"cyto3\"); models.Cellpose(model_type=\"nuclei\")"]

from base
copy --from=py-base /root/.local/share/uv /root/.local/share/uv
copy --from=py-base /.venv /.venv
env PATH=/.venv/bin:$PATH

copy --from=py-base /root/.cellpose/models /root/.cellpose/models
run ["chmod", "-R", "777", "/root/.cellpose/models"]

copy src/*.py /workdir/
workdir /workdir
