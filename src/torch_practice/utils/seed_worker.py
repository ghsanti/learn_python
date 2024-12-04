"""Utilities."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_worker(_: int) -> None:
  """Seed for torch dataloader."""
  logger.debug("setting seeds for dataloader.")
  worker_seed = torch.initial_seed() % 2**32
  np.random.default_rng(worker_seed)
  random.seed(worker_seed)
