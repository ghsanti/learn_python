![3.10|3.11](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
![devtools](https://img.shields.io/badge/astral-uv_ruff-orange)
![test](https://img.shields.io/badge/test-pytest-blue)
![precommit](https://img.shields.io/badge/pre_commit-blue)

![main](https://img.shields.io/badge/version-0.0.1-red)

Simple PyTorch AutoEncoder to play with.

## Set Up

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

### Elsewhere

As an example:

```bash
python -m venv
source .venv/bin/activate
pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice
```

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
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra cpu # or cu124
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
