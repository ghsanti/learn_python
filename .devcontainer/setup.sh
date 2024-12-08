pip install --upgrade pip
pip install uv
uv venv
source .venv/bin/activate
uv python install 3.10 # to be certain
uv sync
# can't add this to pyproject, too hard to conf diff hardware.
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
