"""Config object definition."""

import typing
from collections.abc import Callable
from typing import Literal, TypedDict

import torch

if typing.TYPE_CHECKING:
  from torch.utils.data import DataLoader
  from torchvision.datasets import CIFAR10

  CIFAR = DataLoader[CIFAR10]
  _LogLevel = Literal["DEBUG", "INFO", "WARN", "CRITICAL"]
else:
  CIFAR = None
  _LogLevel = None

LossModeType = Literal[
  "min",
  "max",
]
SaveModeType = Literal["all", "better"]


class DAEConfig(TypedDict):
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
  loss_mode: LossModeType
  # min for minimisation (like MSE),
  # max for maximisation (like accuracy).
  save: SaveModeType | None
  # all: all models
  # better: if improves wrt previous
  # None: no saving.
  save_every: int  # this saves only every `int` epochs.
  # (compounds with "better" if set.)
  save_basedir: str  # save within, using subdirectory with the timestamp,
  # this is for safety. (avoids overwriting, reusing some dir, etc.)

  # general configuration
  layers: int  # Number of layers in the encoder/decoder.
  growth: float  # Growth factor for channels across layers.
  batch_size: int  # critical hyperparameter.
  input_size: tuple[
    int,
    int,
    int,
  ]  # channels, height, width
  lr: float  # learning rate
  epochs: int

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
