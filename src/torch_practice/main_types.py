"""Configure Runtime, Arch, Saving etc."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

CIFAR = DataLoader[CIFAR10]

# Saving Types.
SaveAtType = Literal["all", "improve"]
# torchscript unsupported bc they do not allow basic typing.
SaveModeType = Literal["state_dict", "full_model"]
Extensions = Literal[".pth", ".pt", ".tar"]


class SaverBaseArgs(TypedDict):
  basedir: str | Path
  save_every: int
  save_mode: SaveModeType
  save_at: SaveAtType


# Logging Levels
_LogLevel = Literal["DEBUG", "INFO", "WARN", "CRITICAL"]

# Is loss minimised or maximised.
LossModeType = Literal[
  "min",
  "max",
]


# Architecture configuration
class DAEConfig(TypedDict):
  input_size: tuple[
    int,
    int,
    int,
  ]  # channels, height, width
  layers: int  # Number of layers in the encoder/decoder.
  growth: float  # Growth factor for channels across layers.

  # conv layers
  init_out_channels: int  # initial output channels (1st conv.)
  c_kernel: int  # Kernel size for convolution layers.
  c_stride: int  # Stride for convolution layers.
  c_activ: Callable[[torch.Tensor], torch.Tensor]  # activation function

  # dropout layers
  use_dropout2d: bool  # for convolutions
  dropout2d_rate: float
  use_dropout_latent: bool
  dropout_rate_latent: float  # dense layer before and after the latent vec.

  # pool layers
  use_pool: bool
  p_kernel: int  # Kernel size for pooling layers.
  p_stride: int  # Stride for pooling layers.

  # latent vector
  latent_dimension: int
  dense_activ: Callable[[torch.Tensor], torch.Tensor]  # activation function


class RunConfig(TypedDict):
  """Configuration Dictionary for DAE params.

  Note: BatchNorm always runs, so there isn't a switch.
  """

  # runtime config
  seed: int | None  # if an int, uses `torch.set_manual(seed)`
  log_level: _LogLevel
  gradient_log: bool  # log the max abs value for each gradient in the net.
  data_dir: str
  # fraction on train, fraction on test (must add to 1)
  prob_split: tuple[float, float]
  # n_workers for dataloaders
  n_workers: int
  loss_mode: LossModeType  # min=minimisation, max=maximisation.

  # Hyperparameters
  batch_size: int  # critical hyperparameter.
  epochs: int
  lr: float  # learning rate

  # None won't save anything.
  saver: SaverBaseArgs

  # Architecture definition
  arch: DAEConfig
