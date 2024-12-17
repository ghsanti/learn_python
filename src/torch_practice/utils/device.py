"""Get the device to use."""

import torch


def get_device_name() -> str:
  """Get the device to use."""
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.mps.is_available():
    device = "mps"
  return device
