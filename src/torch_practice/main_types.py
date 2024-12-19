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
  save_mode: SaveModeType | None
  save_at: SaveAtType | None


_LogLevel = Literal["DEBUG", "INFO", "WARN", "CRITICAL"]


class LoggerBaseArgs(TypedDict):
  log_level: _LogLevel
  gradients: bool
  network_graph: bool
  tboard_dir: str | None


LossModeType = Literal["min", "max"]
"""Is loss minimised or maximised."""

CHW = tuple[int, int, int]
"""Channels, Height, Width tuple."""


# Architecture configuration
class DAEConfig(TypedDict):
  input_size: CHW
  """`(channels, height, width)`"""
  layers: int
  """Number of layers in the encoder/decoder."""
  growth: float
  """Growth factor for channels across layers."""
  # conv layers
  init_out_channels: int
  """initial output channels (1st conv.)"""
  c_kernel: int
  """Kernel size for convolution layers."""
  c_stride: int
  """Stride for convolution layers."""
  c_activ: Callable[[torch.Tensor], torch.Tensor]
  """activation function"""
  # dropout layers
  dropout2d_rate: float | None
  dropout_rate_latent: float | None
  """dense layer before and after the latent vec."""
  # pool layers
  use_pool: bool
  p_kernel: int
  """Kernel size for pooling layers."""
  p_stride: int
  """Stride for pooling layers."""
  latent_dimension: int
  """Dimensionality of the encoded vector (encoder output.)"""
  dense_activ: Callable[[torch.Tensor], torch.Tensor]
  """activation function"""


class RunConfig(TypedDict):
  """Configuration Dictionary for DAE params.

  Note: BatchNorm always runs, so there isn't a switch.
  """

  # runtime config
  seed: int | None
  """if an int, uses `torch.set_manual(seed)`"""
  logger: LoggerBaseArgs
  """The logger is tied to the RunConfig, but these are its init args."""
  data_dir: str
  """Where to search for the datasets, or it downloads it therein."""
  prob_split: tuple[float, float]
  """fraction on train, fraction on test (must add to 1)"""
  n_workers: int
  """n_workers for dataloaders"""
  autocast_dtype: torch.dtype | None
  """Datatypes for autocast: torch.float16 | torch.bfloat16 | None.
  None uses the default (float32)."""
  loss_mode: LossModeType
  """min or max"""
  # Hyperparameters
  batch_size: int
  """How the dataloader batches the data, and it's passed to training."""
  epochs: int
  """Total number of epochs."""
  lr: float
  """Learning rate"""
  patience: int
  """How many epochs to wait before reducing learning rate"""
  saver: SaverBaseArgs
  """Base arguments for `Save.__init__`. `None` won't use it."""
  arch: DAEConfig
  """Architecture definition"""
