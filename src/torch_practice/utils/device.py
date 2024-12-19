"""Get the device to use."""

from typing import Literal

import torch


def get_device_name() -> Literal["mps", "cuda", "cpu"]:
  """Get the device to use."""
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.mps.is_available():
    device = "mps"
  return device
