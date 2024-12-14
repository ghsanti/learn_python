![3.10|3.11](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
![devtools](https://img.shields.io/badge/astral-uv_ruff-orange)
![test](https://img.shields.io/badge/test-pytest-blue)
![precommit](https://img.shields.io/badge/pre_commit-blue)

![main](https://img.shields.io/badge/version-0.0.1-red)

Simple PyTorch AutoEncoder to play with.

## Set Up

Important note. Repo can be created from devcontainer (codespace or vscode devcontainers), but not on Mac M1s and such. That is because it'd be creating Linux/aarm64 and no PyTorch wheel is available.

You can also spin up a Python 3.10 Docker Image and use it (for Mac Silicon this may cause installation issues though.)

### Google Colab

Check out simple examples in the [Notebooks](./notebooks/).

### Elsewhere

For new project:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Then run

```bash
uv venv --python 3.10
source .venv/bin/activate
uv add torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice@dev
```

Or from pip:

```bash
python3 -m pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice@dev
```

For GPUs, use the extra `[cu124]` instead of `[cpu]`.

For other systems, `cpu` will works (including Apple Silicon like M1s)



One then can run it:
```bash
python -m torch_practice.simple_train
```

For custom configurations, write a simple script:

```python
from torch_practice.simple_train import train
from torch_practice.default_config import default_config

config = default_config()
config["n_workers"] = 3

# then train it.
train(config)
```


This package installs **torch+cpu** by default. For other hardware please install [torch from the matrix versions.](https://pytorch.org/get-started/locally/)


## Configuration

The "blueprint" is in the [DAEConfig, in this file.](./src/torch_practice/main_types.py)

## Reproducibility

<details>
<summary>basic practices</summary>
From the [docs](https://pytorch.org/docs/stable/notes/randomness.html):

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.

To control the sources of randomness one can pass a seed to the configuration dictionary. This controls some ops and dataloading.
</details>

## Dev

<details>
<summary>simple steps here</summary>
1. Fork
2. Clone your fork and run
```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync --all-extras
# non-cpu users need extra torch installs.
```

Checking out to a Codespace it installs everything. Activate the venv using:

```bash
source .venv/bin/activate
```

* In both cases, remember to select the `.venv` python-interpreter in VSCode.
* Use absolute imports.

</details>

## Build

```python
uv pip install --upgrade build
uv build
```
