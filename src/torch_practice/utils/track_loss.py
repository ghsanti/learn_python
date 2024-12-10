"""Utilities to track loss."""

import logging

from torch_practice.main_types import LossModeType

logger = logging.getLogger(__package__)


def loss_improved(
  best: float | None,
  new: float,
  mode: LossModeType,
) -> bool:
  """Compare current and previous loss."""
  if best is None:  # first iteration.
    best = float("inf") if mode == "min" else float("-inf")

  result = (mode == "min" and new < best) or (mode == "max" and new > best)
  log(best, new, result=result)

  return result


def log(best: float, new: float, *, result: bool) -> None:
  """Debug info about loss behaviour."""
  res_msg = "better" if result else "worse"
  msg = f"New ({new}) {res_msg} than Old ({best})."
  logger.debug(msg)
