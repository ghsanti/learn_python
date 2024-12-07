"""Saving/Loading utilities.

Pass the whole "config" makes it more robust.
"""

import logging
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

  Parses the filenames and gets best name from best loss.
  """
  best_loss = float("-inf") if mode == "max" else float("inf")
  best_name = Path()

  path_names = dirname.glob("*.pth")  # may be empty raising error later on.
  try:  # so we "try"
    for pathname in path_names:
      loss = float(pathname.stem.split("_")[-1])
      improved = loss_improved(best_loss, loss, mode)
      if improved:
        best_loss = loss
        best_name = pathname

  except StopIteration as sti:
    msg = "No model matches the pattern '*.pth'"
    raise FileNotFoundError(msg) from sti

  if best_name.name == "":
    msg = "Could not parse loss from filename."
    raise LossesNotFoundError(msg)
  return dirname / best_name


def load_model(
  net: DynamicAE,
  save_dir: Path,
  filename: str | None = None,
  mode: LossModeType | None = None,
) -> NamedTuple:
  """Set the state dict to the model instance.

  Args:
    net: instance of the model.
    save_dir: directory to save.
    filename: the filename of the target model.
    mode: the loss modes available.

  Returns:
    Does not return the model, only unexpected keys. The model is mutated.

  The directory it searches on, is the `save_dir` in the config file.

  """
  fullname = None
  if filename is not None:
    fullname = save_dir / filename
  else:  # filename is defined
    if mode is None:
      msg = f"If `filename=None`, `mode` must be defined. Found mode={mode}"
      raise ValueError(msg)
    msg = "`name` is unspecified, finding best model..."
    logger.debug(msg)
    fullname = get_best_path(save_dir, mode)

  if fullname is None:
    msg = "The constructed file-fullname can't be None."
    raise ValueError(msg)
  if not fullname.exists():
    msg = f"File with name {fullname} was not found."
    raise FileNotFoundError(msg)

  msg = f"Loading the state dict into {net.__class__.__name__}"
  logger.info(msg)

  return net.load_state_dict(torch.load(fullname, weights_only=True))


def make_savedir(basedir: str) -> Path:
  """Create timestamped directory within basedir.

  This function will error if the new directory already exists.
  """
  date = datetime.now(timezone.utc).strftime("%Y_%m_%d_T%H_%M_%SZ")
  savedir = Path(basedir) / date
  savedir = savedir.expanduser()
  savedir.mkdir(parents=True, exist_ok=False)
  return savedir


class LossesNotFoundError(Exception):
  pass
