![3.10|3.11](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
![devtools](https://img.shields.io/badge/astral-uv_ruff-orange)
![test](https://img.shields.io/badge/test-pytest-blue)
![precommit](https://img.shields.io/badge/pre_commit-blue)

![main](https://img.shields.io/badge/version-0.0.1-red)

Simple PyTorch AutoEncoder to play with.

## Set Up

For Colab use:

```bash
!pip3 install torch_practice[cu124]@git+https://github.com/ghsanti/torch_practice
```
That's for CUDA. For CPU replace `[cu124]` for `[cpu]`. That's all you need.

You can check out simple examples in the [Notebooks](./notebooks/).

----------

<details>
<summary>
Important remarks for devcontainers.
</summary>

Repo can be used from devcontainers which is highly recommended. This package does not
remove files, but it does write out:
* timestamped folders for checkpoints (optionally). 
* and a datafolder for the dataset downloaded directly through PyTorch (no custom code.)

The default locations are all within the configuration file linked further down.

The container should set up any CPU system just fine.

* It won't install any GPU libraries, nor will allow use of MPS which is a MacOS feature,
and you'll be running Linux (Debian with Python 3.10)
* It should be possible to just install the GPU version from within, but this is
untested.
* To run the notebooks in VSCode, you may need `uv sync --extra cpu --extra ipynb`
* Or from pip `pip install "ipykernel>6.29"`
</details>

### Non-Colab install

For new project:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Then run

```bash
uv venv --python 3.10
source .venv/bin/activate
uv add torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice@dev
```

Or from pip (but it needs Python 3.10 currently.):

```bash
python3 -m pip install torch_practice[cpu]@git+https://github.com/ghsanti/torch_practice@dev
```

For GPUs, use the extra `[cu124]` instead of `[cpu]`.

For other systems, `cpu` will work (incl. Apple Silicon like M1s)

If you want a pre release of pytorch, you can try removing `[cpu]`

And using something like (from within the virtual environment):


```bash
 python -m ensurepip
 python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
 ```

 Which is taken from [the nightly tab of torch start locally.](https://pytorch.org/get-started/locally/)


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
