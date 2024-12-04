"""Utilities."""

import pickle
from pathlib import Path

from ray.train import get_checkpoint
from torch import nn, optim


def reload_state(net: nn.Module, optimizer: optim.Optimizer) -> int:
  """Reload the Network state from checkpoint."""
  checkpoint = get_checkpoint()
  if checkpoint:
    with checkpoint.as_directory() as checkpoint_dir:
      data_path = Path(checkpoint_dir) / "data.pkl"
      with Path.open(data_path, "rb") as fp:
        checkpoint_state = pickle.load(fp)
      start_epoch = checkpoint_state["epoch"]
      net.load_state_dict(checkpoint_state["net_state_dict"])
      optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
  else:
    start_epoch = 0

  return start_epoch
