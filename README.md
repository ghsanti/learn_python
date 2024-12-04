![3.10|3.11](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
![devtools](https://img.shields.io/badge/astral-uv_ruff-orange)
![test](https://img.shields.io/badge/test-pytest-blue)
![precommit](https://img.shields.io/badge/pre_commit-blue)

![main](https://img.shields.io/badge/version-0.0.1-red)

Mini-project to learn:

- Ray-Tune for Hyperparameter tuning
- PyTorch to write some nets
- Learn profiling (`torch.profiling`) and tensorboard vis.
- Python packaging, distribution and publishing
- Using some actions and devtools
- Using Readme shields
- Save and Convert to ONNX to use on the web.
- Use Accelerate to delegate device selection code.

## Set Up


### Google Colab

```bash
!pip install git+https://github.com/ghsanti/torch_practice -q
```

Then,

1. Import Accelerate
2. Pass the training function


```python
from functools import partial

from accelerate import notebook_launcher

from torch_practice.simple_train import train
from torch_practice.default_config import default_config

config = default_config()
config["n_workers"] = 3

# then train it.
train(config)

notebook_launcher(partial(train, config))
```

It should use all the power, but may error in some settings (in TPU, for example.)

### Elsewhere
For Unix, on CPU using `uv`, with the `.venv` activated:

For the simplest case (CPU), you can run a training with:  

```python
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

For more complex hardware configurations, it should suffice to:

1. Activate the `.venv`
2. Use `accelerate config` configurate the device.
3. Train using `accelerate launch path/to/train.py`

And install whichever torch is needed [following the matrix versions.](https://pytorch.org/get-started/locally/) 


## Configuration

The "blueprint" is in the [DAEConfig, in this file.](./src/torch_practice/main_types.py)

## Reproducibility

From the [docs](https://pytorch.org/docs/stable/notes/randomness.html):

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.

To control the sources of randomness one can pass a seed to the configuration dictionary. This controls some ops and dataloading.

You can add extra strategies if it's needed.

## Dev

<details>
<summary>simple steps here</summary>
1. Fork
2. Clone your fork and run
```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync
# install torch with pip as detailed at the top
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

It's easier to checkout to a Codespace. It installs everything  for you, just activate the venv using:
```bash
source .venv/bin/activate
```

In both cases, remember to select the `.venv` python-interpreter in VSCode.

Files with "\_\_main\_\_" which can be executed as scripts need to use absolute imports (`from torch_practice import xyz`). The rest can use relative (`from .axes import xyz`).

</details>

## Build

```python
uv pip install --upgrade build
uv build
```

## PyTorch settings

`autocast` isn't worth even trying, it makes things slower. One can check even unresolved issues. Ofc, this is for CPU, for other devices it probably changes.
