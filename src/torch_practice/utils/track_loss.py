"""Utilities to track loss."""

import logging

from torch_practice.main_types import DAEConfig

logger = logging.getLogger(__package__)


def loss_improved(
  best: float | None,
  new: float,
  config: DAEConfig,
) -> bool:
  """Compare current and previous loss."""
  mode = config.get("loss_mode")

  if best is None:  # first iteration.
    logger.debug("Setting `best` loss because it found None.")
    best = float("inf") if mode == "min" else float("-inf")

  return (mode == "min" and new < best) or (mode == "max" and new > best)
