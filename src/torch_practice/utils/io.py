"""Saving/Loading utilities.

Pass the whole "config" makes it more robust.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import torch

from torch_practice.main_types import LossModeType
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def get_best_path(dirname: Path, mode: LossModeType) -> Path:
  """Find best model if filename is not specified by user.

  The function uses a regex/pattern to find a float in the filename.
  If it doesn't, it throws a "LossNotFoundError."
  """
  # match files like `abc_0.124.pth` (or `.pt`)
  pattern = r".*_(\d+\.\d+)\.pth?$"
  best_loss = None  # None is taken care of within loss_improved.
  best_name = None

  for file in dirname.iterdir():
    if file.is_file():
      match = re.match(pattern, file.name)
      logger.debug("Found model file.")
      if match:
        msg = f"Found match from regex ({match.group(1)})."
        logger.debug(msg)
        loss = float(match.group(1))
        improved = loss_improved(best_loss, loss, mode)
        if improved:
          best_loss = loss
          best_name = file

  if best_name is None:
    msg = f"Could not parse loss from filenames in dir: {dirname}."
    raise LossNotFoundError(msg)
  msg = f"Best filename-model found {best_name}"
  logger.debug(msg)
  return best_name  # full path


def load_model_from_mode(
  net: DynamicAE,
  save_dir: Path,
  mode: LossModeType,
) -> NamedTuple:
  """Set the state dict to the model instance.

  Args:
    net: instance of the model.
    save_dir: directory it searches on.
    mode: the loss modes available.

  Returns:
    Does not return the model, only unexpected keys. The model is mutated.

  """
  logger.debug("finding best model...")
  fullname = get_best_path(save_dir, mode)

  if not fullname.exists():
    msg = f"File with name {fullname} was not found."
    raise FileNotFoundError(msg)

  msg = f"Loading the state dict into {net.__class__.__name__}"
  logger.info(msg)

  return net.load_state_dict(torch.load(fullname, weights_only=True))


def load_model_from_filepath(
  net: DynamicAE,
  path_to_model: Path,
) -> NamedTuple:
  """Set the state dict to the model instance.

  Args:
    net: instance of the model.
    path_to_model: path to the saved `.pth` file.

  Returns:
    Does not return the model, only unexpected keys. The model is mutated.

  """
  return net.load_state_dict(
    torch.load(path_to_model.resolve(), weights_only=True),
  )


def make_savedir(basedir: str) -> Path:
  """Create timestamped directory within basedir.

  This function will error if the new directory already exists.
  """
  date = datetime.now(timezone.utc).strftime("%Y_%m_%d_T%H_%M_%SZ")
  savedir = Path(basedir) / date
  savedir = savedir.expanduser()
  savedir.mkdir(parents=True, exist_ok=False)
  return savedir


class LossNotFoundError(Exception):
  pass
