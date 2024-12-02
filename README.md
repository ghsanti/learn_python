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

## Set Up

Recommended to do it on Colab first:

```bash
!pip install git+https://github.com/ghsanti/torch_practice -q
```

Because PyTorch is quite hard to install for each version, you need to install it manually. [Check their matrix here.](https://pytorch.org/get-started/locally/)

For Linux, on CPU using `uv`, with the `.venv` activated:

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

On Colab CPU probably just remove `uv`.

For Colab GPUs, this should work (check the CUDA version.):

```bash
MY_CUDA=cu118
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/${MY_CUDA}
```


### Speed up

<details>
<summary>
Speed-up for **XLA** devices
</summary>

You can run it with XLA devices (both TPUs and GPUs) just needs an extra package:

On colab just use (for GPUs):

```bash
!pip install torch==2.5.1 torch_xla==2.5.1 https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla_cuda_plugin-2.5.1-py3-none-any.whl -q
```

For TPUs it should be ready to use, but otherwise run:


```bash
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

Or check the `torch_xla` package.

</details>

## Run
Then:

```python
python -m torch_practice.simple_train
```

Should start training with the default configuration.

For custom configurations, write a simple script:

```python
from torch_practice.simple_train import train
from torch_practice.default_config import default_config

# change config, for example:
# autocompletion will show you available properties.
default_config["n_workers"] = 3

# then train it.
train(default_config)
```

<details>

<summary>
Config object "blueprint":
</summary>

```python
class DAEConfig(TypedDict):
  """Configuration Dictionary for DAE params.

  Note: BatchNorm always runs, so there isn't a switch.
  """

  # runtime config
  seed: int | None  # if an int, uses `torch.set_manual(seed)`
  log_level: _LogLevel
  data_dir: str
  # fraction on train, fraction on test (must add to 1)
  prob_split: tuple[float, float]
  # n_workers for dataloaders
  n_workers: int

  # general configuration
  layers: int  # Number of layers in the encoder/decoder.
  growth: float  # Growth factor for channels across layers.
  in_channels: int  # Number of input channels (e.g., 3 for RGB images).
  lr: float  # learning rate
  batch_size: int  # critical hyperparameter.
  clip_gradient_norm: bool
  clip_gradient_value: bool
  epochs: int

  # conv layers
  init_out_channels: int  # initial output channels (1st conv.)
  c_kernel: int  # Kernel size for convolution layers.
  c_stride: int  # Stride for convolution layers.
  c_activ: Callable  # activation function

  # dropout layers
  use_dropout: bool
  dropout_rate: float

  # pool layers
  use_pool: bool
  p_kernel: int  # Kernel size for pooling layers.
  p_stride: int  # Stride for pooling layers.

  # latent vector
  latent_dimension: int
  dense_activ: Callable  # activation function

```

</details>

## Reproducibility

From the [docs](https://pytorch.org/docs/stable/notes/randomness.html):

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.

To control the sources of randomness one can pass a seed to the configuration dictionary. This controls some ops and dataloading.

You can add extra strategies if it's needed.

## Packaging

<details>
<summary>
Open here
</summary>
This project uses `Python>=3.10`

It uses:

- Project Management: [astral/uv](https://github.com/astral-sh/uv)
  One needs to activate the _.venv_ as well.

  After activating, it's good to update _pip_ and _build_

```python
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
```

- Lint: [astral/ruff](https://github.com/astral-sh/ruff)
  Own separate config file.
- Type-Check: [pyright](https://github.com/microsoft/pyright)
  Own separate config file. Install using _[nodejs]_ extra.
- [pre-commit](https://pre-commit.com/)
  Run `precommit install` after installing.
- testing: [pytest](https://docs.pytest.org/)
- pyproject: python project metadata.
- Github Actions: integration for quality checks, PyPI publishing, and documentation publishing.
- Docker for Codespaces integration (_.devcontainer_)

The project itself is a simple neural network in Pytorch, with Hyperparameter tuning using Ray Tune. It uses code from Pytorch examples.

For distributing packages one needs to:

1. Build it (uses _hatch_ in this repo.) Setuptools, Flitch, PDM are other build backends.

One can then use any build frontend (like _build_), they will search for the build backend and create the distribution files.

In this case, the file has:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

To show which backend to use. The frontend installs _hatch_ if needed.

2. Have the right project structure and metadata (_pyproject.toml_) 3. Publish it on PyPI (publicly or privately.)

Key is to have _src/project_name_ and the name of the package in the _pyproject.toml_ file.

Also have \_\_\_init.py\_\_\_ files in each subdirectory. **This allows the user to import the directory as if it were a module.**

The _tests_ folder is outside the source folder, and won't be distributed. Other files outside source will be incl. in the source distribution (_tar.gz_ compression.)

Lastly one needs _Readme.md_ and _LICENSE_

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
