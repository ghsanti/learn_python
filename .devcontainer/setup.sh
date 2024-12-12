pip install --upgrade pip
pip install uv
uv venv
source .venv/bin/activate
uv python install 3.10 # to be certain
uv sync --extra cpu  # dev deps. are synced by default.
