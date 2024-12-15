#!/usr/bin/env sh
pip3 install --upgrade pip
pip3 install -U uv
if [ -d "/.venv" ]; do
    rm -rf .venv
fi
uv venv --python 3.10 --verbose
# `source` not available from scripts
. .venv/bin/activate
# for Apple Silicon, fallsback to PyPI 
uv sync --extra cpu --extra ipynb
# dev deps. are synced by default.
