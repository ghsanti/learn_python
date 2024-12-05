"""Utilities to track loss."""

import logging

from torch_practice.main_types import DAEConfig

logger = logging.getLogger(__package__)


def loss_improved(
  best: float,
  new: float,
  config: DAEConfig,
) -> bool:
  """Compare current and previous loss."""
  mode = config.get("loss_mode")

  result = (mode == "min" and new < best) or (mode == "max" and new > best)
  if result is True:
    logger.info("Old loss: %s - New loss: %s", best, new)
  return result
