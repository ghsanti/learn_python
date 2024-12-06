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
