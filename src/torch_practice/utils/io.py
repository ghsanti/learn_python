"""Saving/Loading utilities.

Pass the whole "config" makes it more robust.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def remove_old_best(dirname: Path, glob: str = "best_*.pth") -> None:
  """Remove one best checkpoint."""
  try:
    old_best = next(dirname.glob(glob))
    logger.info("removing file %s", old_best)
    old_best.unlink(missing_ok=True)
  except StopIteration:
    logger.info("No `best_*` file found to remove at %s", dirname)


def save_model(net: nn.Module, config: DAEConfig, name: str) -> None:
  """Save the model state dict."""
  dirname = Path(config.get("save_dir")).expanduser()
  dirname.mkdir(parents=True, exist_ok=True)
  if config.get("save") == "best_only":
    remove_old_best(dirname)
    name = "best_" + name

  suffix = "" if name.endswith(("pth", "pt")) else "pth"
  fullname = dirname / Path(name + suffix)
  torch.save(net.state_dict(), fullname)
  logger.info('Saved "%s" to %s', config.get("save"), str(fullname))


def get_best_path(dirname: Path, config: DAEConfig) -> Path:
  """Find best model if filename is not specified by user.

  Parses the filenames and gets best name from best loss.
  """
  mode = config.get("loss_mode")
  best_loss = float("-inf") if mode == "max" else float("inf")
  best_name = Path()

  path_names = dirname.glob("*.pth")  # may be empty raising error later on.
  # so we "try"
  try:
    for pathname in path_names:
      loss = float(pathname.stem.split("_")[-1])
      improved = loss_improved(best_loss, loss, config)
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
  config: DAEConfig,
  name: str | None = None,
) -> NamedTuple:
  """Set the state dict to the model instance.

  Args:
    net: instance of the model
    config: configuration class
    name: name within the `save_dir`. If `None`, it will load the best result
    according to the loss value. It's best practice to pass a name here.

  Returns:
    Does not return the model, only unexpected keys. The model is mutated.

  The directory it searches on, is the `save_dir` in the config file.

  """
  dirname = Path(config.get("save_dir")).expanduser()
  fullname = None
  if name is not None:
    fullname = dirname / Path(name)
  else:
    msg = "`name` is unspecified, finding best model..."
    logger.debug(msg)
    fullname = get_best_path(dirname, config)

  if fullname is None:
    msg = "The constructed file-fullname can't be None."
    raise ValueError(msg)
  if not fullname.exists():
    msg = f"File with name {fullname} was not found."
    raise FileNotFoundError(msg)

  msg = f"Loading the state dict into {net.__class__.__name__}"
  logger.info(msg)

  return net.load_state_dict(torch.load(fullname, weights_only=True))


class LossesNotFoundError(Exception):
  pass
