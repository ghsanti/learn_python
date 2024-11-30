"""Utilities."""

import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from ray.train import get_checkpoint
from torch import nn, optim

from torch_practice.main_types import CIFAR

logger = logging.getLogger(__name__)


def to_device_available(net: nn.Module) -> str:
  """Send to device (GPU/MPS) if available.

  Returns
  -------
    device: where it was sent to.

  """
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.device_count() > 1:
      logger.info("Trying GPU in parallel.")
      net = nn.DataParallel(net)
  elif torch.mps.is_available():
    device = "mps"
  net = net.to(device)
  return device


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


def test_accuracy(
  net: nn.Module,
  testloader: CIFAR,
  device: str = "cpu",
) -> float:
  """Test accuracy."""
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return correct / total


def seed_worker(_: int) -> None:
  """Seed for torch dataloader."""
  worker_seed = torch.initial_seed() % 2**32
  np.random.default_rng(worker_seed)
  random.seed(worker_seed)
