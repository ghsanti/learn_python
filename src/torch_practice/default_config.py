"""Runtime default configuration."""

import torch

from torch_practice.utils.io import Save

from .main_types import DAEConfig


def default_config(saver: Save) -> DAEConfig:
  """Full default configuration.

  Args:
  saver: an instance that configures how and when to save a model.

  """
  return {
    # general runtime
    "seed": None,
    "log_level": "DEBUG",
    "gradient_log": False,
    "data_dir": "./data",
    "epochs": 10,
    "prob_split": (0.8, 0.2),
    "n_workers": 2,
    "loss_mode": "min",
    # save handler instance.
    "saver": saver,
    # architecture
    "growth": 2,
    "init_out_channels": 6,
    "layers": 3,
    "lr": 0.001,
    "input_size": (3, 32, 32),
    "batch_size": 12,
    # convolution
    "c_kernel": 2,
    "c_stride": 1,
    "c_activ": torch.nn.functional.leaky_relu,
    # pool
    "use_pool": False,
    "p_kernel": 2,
    "p_stride": 2,
    # dropout
    "use_dropout2d": True,
    "dropout2d_rate": 0.3,
    "dropout_rate_latent": 0.3,
    "use_dropout_latent": False,
    # dense
    "latent_dimension": 96,
    "dense_activ": torch.nn.functional.leaky_relu,
  }
