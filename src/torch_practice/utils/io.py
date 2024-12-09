"""Saving/Loading utilities.

Pass the whole "config" makes it more robust.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

import torch

from torch_practice.main_types import LossModeType
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


SaveAtType = Literal["all", "better"] | None
# in both cases, it's standard state dict saved, not torchscript
# you can load and convert it if necessary.
SaveModeType = Literal["inference", "training"] | None


class Save:
  def __init__(
    self,
    basedir: str | Path,
    save_every: int = 3,
    save_mode: SaveModeType = "inference",
    save_at: SaveAtType = "better",
  ) -> None:
    """Handle model and checkpoint saving.

    Args:
        basedir: path to the basedirectory. It will be appended a date.
        save_every: saves every this number of epochs. Must be >0.
        save_mode: save for "training" or "inference".
        save_at: under which circumstances to save.

    """
    if not isinstance(save_every, int) or save_every < 1:
      msg = f"'save_every' must be a natural number. Found {type(save_every)}"
      raise ValueError(msg)
    if save_at not in ["always", "improves", None]:
      msg = f"Options: 'always', 'improves' and None. Received {save_at}."
      raise ValueError(msg)
    self._basedir = basedir
    self.dirname = self.make_savedir()
    self.mode = save_mode
    self.every = save_every
    self.at = save_at

  # basic utilities for homogeneity.
  def save_time(self, epoch: int) -> bool:
    """Check if it is time to save according to configuration."""
    return ((epoch + 1) % self.every) == 0

  def make_filepath(self, epoch: int, loss: float) -> Path:
    """Make filepath using epoch and loss value."""
    return self.dirname / f"{epoch}_{loss:.3f}.pth"

  def make_savedir(self) -> Path:
    """Create timestamped directory within basedir.

    This function will error if the new directory already exists.
    """
    date = datetime.now(timezone.utc).strftime("%Y_%m_%d_T%H_%M_%SZ")
    savedir = Path(self._basedir) / date
    savedir = savedir.expanduser()
    savedir.mkdir(parents=True, exist_ok=False)
    return savedir

  def save_inference(self, net: DynamicAE, epoch: int, loss: float) -> None:
    """Save state_dict of model-only.

    This is useful to load for inference, not for retraining.
    """
    full_name = self.make_filepath(epoch, loss)
    torch.save(net.state_dict(), str(full_name))
    logger.info("Saved to %s", str(full_name))

  def save_checkpoint(
    self,
    net: DynamicAE,
    epoch: int,
    loss: object,
    loss_value: float,
    optimizer: torch.optim.Optimizer,
  ) -> None:
    """Save all states.

    Note that here the loss is the loss class.

    """
    full_name = self.make_filepath(epoch, loss_value)
    torch.save(
      {
        "model_state_dict": net.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
      },
      full_name,
    )
    logger.info("Saved checkpoint to %s", str(full_name))


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


def load_checkpoint_from_mode(
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

    In the case of inference, only model's state dict is saved.
    In the case of training, Epoch,Model,Loss,Optimizer are saved (and loaded.)

  """
  logger.debug("finding best model...")
  fullname = get_best_path(save_dir, mode)

  if not fullname.exists():
    msg = f"File with name {fullname} was not found."
    raise FileNotFoundError(msg)

  msg = f"Loading the state dict into {net.__class__.__name__}"
  logger.info(msg)

  return torch.load(fullname, weights_only=True)


def load_inference_model_from_mode(
  net: DynamicAE,
  save_dir: Path,
  mode: LossModeType,
) -> NamedTuple:
  """Set the state dict to the model instance.

  Args:
    net: instance of the model.
    save_dir: directory it searches on.
    savefor: the type of model saved (see SaveFor type.)
    mode: the loss modes available.

  Returns:
    Does not return the model, only unexpected keys. The model is mutated.

    In the case of inference, only model's state dict is saved.
    In the case of training, Epoch,Model,Loss,Optimizer are saved (and loaded.)

  """
  logger.debug("finding best model...")
  fullname = get_best_path(save_dir, mode)

  if not fullname.exists():
    msg = f"File with name {fullname} was not found."
    raise FileNotFoundError(msg)

  msg = f"Loading the state dict into {net.__class__.__name__}"
  logger.info(msg)

  checkpoint = torch.load(fullname, weights_only=True)
  return net.load_state_dict(checkpoint)


def load_checkpoint_from_filename(
  path_to_model: Path,
) -> NamedTuple:
  """Checkpoint loading.

  Args:
    path_to_model: path to the saved `.pth` file.

  Returns:
    Named tuple of stateful items (Model, Loss, Epoch, Optimizer.)

  """
  return torch.load(path_to_model.resolve(), weights_only=True)


class LossNotFoundError(Exception):
  pass
