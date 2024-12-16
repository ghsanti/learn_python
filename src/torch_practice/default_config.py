"""Runtime default configuration."""

import torch

from torch_practice.main_types import RunConfig


def default_config() -> RunConfig:
  """Create a default configuration."""
  return {
    # general runtime
    "seed": None,
    "log_level": "DEBUG",
    "gradient_log": False,
    "data_dir": "./data",
    "prob_split": (0.8, 0.2),
    "n_workers": 2,
    "loss_mode": "min",
    # HyperParameters (also the architecture to an extent.)
    "batch_size": 12,
    "autocast_dtype": torch.bfloat16,
    "print_network_graph": True,
    "lr": 0.001,
    "epochs": 10,
    "saver": {  # replace dict with None for no saving!
      "basedir": "checkpoints",
      "save_every": 3,
      "save_mode": "state_dict",
      "save_at": "improve",
    },
    "arch": {
      # architecture
      "growth": 2,
      "init_out_channels": 6,
      "layers": 3,
      "input_size": (3, 32, 32),
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
    },
  }
