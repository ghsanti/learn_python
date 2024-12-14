#/usr/bin/env sh
pip3 install --upgrade pip
pip3 install -U uv
uv venv --python 3.10
source .venv/bin/activate
# dev deps. are synced by default.
uv sync --extra cpu
