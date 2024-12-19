"""Utilities for model loading."""

import logging
import re
from pathlib import Path
from re import Match
from typing import TYPE_CHECKING, NamedTuple

import torch

from torch_practice.main_types import LossModeType, SaveModeType
from torch_practice.utils.date_format import (
  DirnameParsingError,
  assert_date_format,
)
from torch_practice.utils.device import get_device_name
from torch_practice.utils.track_loss import loss_improved

if TYPE_CHECKING:
  from torch_practice.nn_arch import DynamicAE
else:
  DynamicAE = None
logger = logging.getLogger(__name__)

# full models are saved in .tar, state dicts in .pth


def load_state_dict(net: DynamicAE, filepath: Path) -> NamedTuple:
  """Load model state from state dict."""
  device = torch.device(get_device_name())
  return net.load_state_dict(
    torch.load(filepath, map_location=torch.device(device), weights_only=True),
  )


def load_full_model(filepath: Path, *, weights_only: bool) -> dict:
  """Load full model."""
  device = torch.device(get_device_name())
  return torch.load(
    filepath,
    map_location=torch.device(device),
    weights_only=weights_only,
  )


class LossNotFoundError(Exception):
  pass


class LossComparisonError(Exception):
  pass


def run_match(filepath: Path, save_mode: SaveModeType) -> Match[str] | None:
  """Check format and extract loss if available."""
  ext = "pth?" if save_mode == "state_dict" else "tar"
  loss_and_mode_pattern = rf".*_(\d+\.\d+)\.{ext}$"
  return re.match(loss_and_mode_pattern, filepath.name)


def extract_improvement(
  filepath: Path,
  save_mode: SaveModeType,
  loss_mode: LossModeType,
  best_loss: float | None,
) -> tuple[Path, float] | None:
  """Check that is a target file using loss-and-mode search regex."""
  match = run_match(filepath, save_mode)
  if bool(match):
    try:
      loss = float(match.group(1))
      improved = loss_improved(best_loss, loss, loss_mode)
      if improved:
        return filepath, loss

    except Exception as err:
      msg = "Error during extraction and comparison of loss."
      raise LossComparisonError(msg) from err
  return None


def get_best_path(
  start_from: Path,
  loss_mode: LossModeType,
  depth: int,
  save_mode: SaveModeType,
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
  best_name, best_loss = None, None

  for file in start_from.iterdir():
    if file.is_file():
      logger.debug("Inspecting filepath: %s", file)
      result = extract_improvement(file, save_mode, loss_mode, best_loss)
      if result is not None:  # no improvement or invalid file format.
        best_name, best_loss = result
    elif file.is_dir():
      logger.debug("Searching in dir: %s", file)
      try:  # only parse timestamped directories.
        assert_date_format(file)
        if depth > 0:
          result = get_best_path(file, loss_mode, depth - 1, save_mode)

          if result is not None and loss_improved(
            best=best_loss,
            new=result[1],
            mode=loss_mode,
          ):
            best_name, best_loss = result
      except DirnameParsingError as err:
        msg = f"Not a timestamped directory {file}. Error {err}"
        logger.debug(msg)
        continue

  if best_name is not None and best_loss is not None:
    return best_name, best_loss  # full path
  return None
