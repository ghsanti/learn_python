![3.10|3.11](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
![devtools](https://img.shields.io/badge/astral-uv_ruff-orange)
![test](https://img.shields.io/badge/test-pytest-blue)
![precommit](https://img.shields.io/badge/pre_commit-blue)

![main](https://img.shields.io/badge/version-0.0.1-red)

Simple PyTorch Image AutoEncoder to play with.

The load utilities will load a model directly to the best detected device (`cuda>mps>cpu`). This is so that loading some other model (say, trained in colab) doesn't error. You can then place it elsewhere.

Note that before training one needs a forward pass `MyNet(random_tensor)` with
`batch=1` to initialize optional layers.

For example, XLA devices seem to error. These are not supported though just because speed up isn't really that much, and needs several modifications in the code. But it's not hard to write a separate training file.

## Available Set Ups

### Google Colab

Check out simple examples in the [Notebooks](./notebooks/).

For GPU use:

```bash
pip install torch_practice[cu124]@git+https://github.com/ghsanti/torch_practice.git
```

For CPU:

```bash
pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice.git
```

### Codespaces and VSCode

A linux-container version is ready to run by just checking out to Codespaces or opening in VSCode.

This may not work out of the box on Windows given X86_64 architecture, but it won't on Darwin (Mac) if it's aarch64/arm64.

There is still a simple option to run it on any system.

### Manually

If you already have a virtual environment, _activate it_ and use:

```bash
# `cpu` or `cuda124` available
pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice
```

As an example, `uv` is recommended.

```bash
python3 -m pip install -U uv
uv venv --python 3.10
source .venv/bin/activate
# For CUDA change [cpu] to [cuda124], for MPS use [cpu] as well. (it will use GPU.)
uv add torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice
# or
# uv pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice
```

## Run

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

You can also use the utilities and configuration to set up your training.
The saving/loading modules have helper functions for that.

## Configuration

The "blueprint" is in the [RunConfig, in this file.](./src/torch_practice/main_types.py)

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
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[cpu]
# uv sync --extra cpu # or cu124
```

Checking out to a Codespace sets all up. Activate the venv using:

```bash
source .venv/bin/activate
```

- In both cases, remember to select the `.venv` python-interpreter in VSCode.
- Use absolute imports.

</details>

## Build

```python
uv pip install --upgrade build
uv build
```
