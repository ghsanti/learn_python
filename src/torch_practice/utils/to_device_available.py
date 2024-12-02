"""Utilities."""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def to_device_available(net: nn.Module) -> tuple[str, object]:
  """Send neural network to device (TPU/GPU/MPS/CPU) if available.

  Returns
  -------
    device: where it was sent to.

  """
  try:
    import torch_xla  # type: ignore (Opt user install)
    import torch_xla.core.xla_model as xm  # type: ignore (Opt user install)

    logger.info("torch_xla found, computations will run on TPU.")
  except ModuleNotFoundError:
    logger.warning("if you want to run in TPU, please install `torch_xla`")
    xm = None

  device = "cpu"
  if xm is not None:
    device = xm.xla_device()
  elif torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.device_count() > 1:
      logger.info("Trying GPU in parallel.")
      net = nn.DataParallel(net)
  elif torch.mps.is_available():
    device = "mps"
  net = net.to(device)
  return device, xm
