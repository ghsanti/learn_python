"""Runtime default configuration."""

import torch

from torch_practice.main_types import RunConfig


def default_config() -> RunConfig:
  """Create a default configuration."""
  return {
    # GENERAL
    "seed": None,
    "data_dir": "data",
    "prob_split": (0.8, 0.2),
    "n_workers": 2,
    "loss_mode": "min",
    # HyperParameters
    "batch_size": 12,
    "autocast_dtype": None,
    "lr": 0.0002,
    "patience": 7,
    "epochs": 10,
    "logger": {
      "gradients": True,
      "log_level": "DEBUG",
      "network_graph": True,
      "tboard_dir": "tboard_logs",
    },
    "saver": {  # `basedir: None` for no saving
      "basedir": "checkpoints",
      "save_every": 3,
      "save_mode": "state_dict",
      "save_at": "improve",
    },
    "arch": {
      # architecture
      "growth": 2,
      "init_out_channels": 12,
      "layers": 3,
      "input_size": (3, 32, 32),
      # convolution
      "c_kernel": 2,
      "c_stride": 2,
      "c_activ": torch.nn.functional.relu,
      # pool
      "use_pool": False,
      "p_kernel": 2,
      "p_stride": 1,
      # dropout
      "dropout2d_rate": 0.2,
      "dropout_rate_latent": 0.2,
      # dense
      "latent_dimension": 72,
      "dense_activ": torch.nn.functional.relu,
    },
  }
