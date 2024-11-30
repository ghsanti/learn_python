import torch

from torch_practice.main_types import DAEConfig


def default_config() -> DAEConfig:
  """Runtime and model default configuration."""
  return {
    # general runtime
    "seed": None,
    "log_level": "DEBUG",
    "data_dir": "./data",
    "epochs": 10,
    "prob_split": (0.8, 0.2),
    "n_workers": 2,
    # general-model
    "growth": 2,
    "in_channels": 3,
    "init_out_channels": 24,
    "layers": 5,
    "lr": 0.001,
    "batch_size": 6,
    "clip_gradient_value": True,
    "clip_gradient_norm": True,
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
