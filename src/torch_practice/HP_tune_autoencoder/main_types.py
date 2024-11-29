"""Config object definition."""

from collections.abc import Callable
from typing import TypedDict

import torch


class DAEConfig(TypedDict):
  """Configuration Dictionary for DAE params.

  Note: BatchNorm always runs, so there isn't a switch.
  """

  # general configuration
  layers: int  # Number of layers in the encoder/decoder.
  growth: float  # Growth factor for channels across layers.
  in_channels: int  # Number of input channels (e.g., 3 for RGB images).
  lr: float  # learning rate
  batch_size: int  # critical hyperparameter.

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


def default_config() -> DAEConfig:
  """Autoencoder default configuration."""
  return {
    # general
    "growth": 2,
    "in_channels": 3,
    "init_out_channels": 24,
    "layers": 5,
    "lr": 0.001,
    "batch_size": 6,
    # convolution
    "c_kernel": 2,
    "c_stride": 1,
    "c_activ": torch.nn.functional.leaky_relu,
    # pool
    "use_pool": False,
    "p_kernel": 2,
    "p_stride": 2,
    # dropout
    "use_dropout": True,
    "dropout_rate": 0.3,
    # dense
    "latent_dimension": 128,
    "dense_activ": torch.nn.functional.leaky_relu,
  }
