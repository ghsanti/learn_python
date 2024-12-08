"""Runtime default configuration."""

import torch

from .main_types import DAEConfig


def default_config() -> DAEConfig:
  """Runtime and model default configuration."""
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
    "save": "better",  # None never saves.
    "save_every": 3,  # 1 saves every epoch
    "save_basedir": "checkpoints",
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


def config_sanity_check(config: DAEConfig) -> None:
  """Check critical configuration keys."""
  image_size = 3  # not channels, but CHW dimensions.
  every = config["save_every"]
  save_mode = config["save"]
  if not isinstance(every, int):
    msg = f"'save_every' must be 'int'. Found {type(every)}"
    raise TypeError(msg)
  if every < 1:
    msg = f"'save_every' must be > 1. Found {config['save_every']}"
    raise ValueError(msg)
  isize = len(config.get("input_size"))
  if isize != image_size:
    msg = f"'input_size' must be of length=3, found {isize}"
    raise ValueError(msg)
  if save_mode not in ["all", "better", None]:
    msg = f"Supported save modes: 'all', 'better' and None. Found {save_mode} "
    raise ValueError(msg)
