"""Utilities for model loading."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from torch_practice.main_types import LossModeType, SaveModeType
from torch_practice.utils.date_format import (
  DirnameParsingError,
  assert_date_format,
)
from torch_practice.utils.track_loss import loss_improved

if TYPE_CHECKING:
  from torch_practice.nn_arch import DynamicAE
else:
  DynamicAE = None
logger = logging.getLogger(__package__)

LOSS_PATTERN = r".*_(\d+\.\d+)\.pth?$"


def load_state_dict(net: DynamicAE, filepath: Path) -> NamedTuple:
  """Load model state from state dict."""
  return net.load_state_dict(torch.load(filepath, weights_only=True))


def load_full_model(filepath: Path, *, weights_only: bool) -> dict:
  """Load full model."""
  return torch.load(filepath, weights_only=weights_only)


class LossNotFoundError(Exception):
  pass


def check_filename_mode(filepath: Path, mode: SaveModeType) -> bool:
  """Check mode from filename."""
  return filepath.name.startswith(mode)


def get_best_path(
  start_from: Path,
  loss_mode: LossModeType,
  depth: int,
  save_mode: SaveModeType | None,
) -> tuple[Path, float] | None:
  """Find best model if filename is not specified by user.

  Args:
    start_from: where to start reading directories from.
    loss_mode: the loss mode (min, max,..)
    depth: how many folders down the tree to look at.
    save_mode: enables a simple filename check to filter models.

  It can be used to pass the name to the `load_model` function.
  The function uses a regex/pattern to find a float in the filename.
  If it doesn't, it throws a "LossNotFoundError."

  """
  # match files like `abc_0.124.pth` (or `.pt`)
  loss_pattern = LOSS_PATTERN
  best_name, best_loss = None, None

  for file in start_from.iterdir():
    match = (
      check_filename_mode(file, save_mode) if save_mode is not None else False
    )
    if file.is_file() and match is not False:
      result = _extract_compare(file, loss_pattern, best_loss, loss_mode)
      if result is not None:
        best_name, best_loss = result
    elif file.is_dir():
      try:  # only parse timestamped directories.
        assert_date_format(file)
        if depth > 0:
          depth -= 1
          result = get_best_path(file, loss_mode, depth, save_mode)
          if result is not None and loss_improved(
            best_loss,
            result[1],
            loss_mode,
          ):
            best_name, best_loss = result
      except DirnameParsingError as err:
        msg = f"Not a timestamped directory {file}. Error {err}"
        logger.debug(msg)
        continue

  if best_name is not None and best_loss is not None:
    return best_name, best_loss  # full path
  return None


def _extract_compare(
  file: Path,
  pattern: str,
  best_loss: float | None,
  mode: LossModeType,
) -> tuple[Path, float] | None:
  """Compare extracted vs best loss.

  The extraction is from the filenames `epoch_loss.pth`.
  """
  # could use stem and different pattern. match as below seems safer.
  match = re.match(pattern, file.name)
  logger.debug("Found model file.")
  if match:
    msg = f"Found match from regex ({match.group(1)})."
    logger.debug(msg)
    loss = float(match.group(1))
    improved = loss_improved(best_loss, loss, mode)
    if improved:
      return file, loss
  return None
