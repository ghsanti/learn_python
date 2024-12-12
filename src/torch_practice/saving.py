"""Handle model saving."""

import logging
from pathlib import Path

import torch

from torch_practice.main_types import SaveModeType, SaverBaseArgs
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.date_format import assert_date_format, make_timestamp

logger = logging.getLogger(__package__)


class Save:
  base_config: SaverBaseArgs

  def __init__(
    self,
    base_config: SaverBaseArgs,
    net: DynamicAE,
    criterion: object,
    optimizer: torch.optim.Optimizer,
  ) -> None:
    """Handle model and checkpoint saving.

      Upon instantiation, it creates the timestamped directory within basedir.

    Args:
        base_config:
          basedir: path to base directory. Time-stamped subdir is created within
          save_every: saves every this number of epochs. Must be >0.
          save_mode: save for "training" or "inference".
          save_at: under which circumstances to save.
        net: the instance of the architecture.
        criterion: the instance of the criterion.
        optimizer: the instance of the optimizer.

    """
    self.mode = base_config["save_mode"]
    self.supported_modes: set[SaveModeType] = {
      "state_dict",
      "full_model",
    }
    self.every = base_config["save_every"]
    self.at = base_config["save_at"]
    self.dirname = self._make_savedir(base_config["basedir"])
    self.user_saving = self.at is not None and self.mode is not None
    self.net = net
    self.criterion = criterion
    self.optim = optimizer

    if self.mode not in self.supported_modes:
      msg = f"{self.mode} not in supported modes ({self.supported_modes})"
      raise ValueError(msg)

  # basic utilities for homogeneity.
  def save_time(self, epoch: int) -> bool:
    """Check if it is time to save according to configuration."""
    return ((epoch + 1) % self.every) == 0

  def make_filepath(self, epoch: int, loss: float) -> Path:
    """Make filepath using mode, epoch and loss."""
    return self.dirname / f"{self.mode}_{epoch}_{loss:.3f}.pth"

  def _make_savedir(self, basedir: str | Path) -> Path:
    """Create timestamped directory within basedir.

    This function will error if the new directory already exists.
    """
    self.basedir = basedir
    savedir = Path(basedir) / make_timestamp()
    savedir = savedir.expanduser()
    assert_date_format(savedir)  # re-check it can be parsed back.
    savedir.mkdir(parents=True, exist_ok=False)
    return savedir

  def save_model(self, epoch: int, loss_value: float) -> Path:
    """Save model at automatically-made path, return saved path."""
    full_name = self.make_filepath(epoch, loss_value)
    match self.mode:
      case "state_dict":
        self.save_state_dict(full_name)
      case "full_model":
        self.save_full_model(full_name, epoch)
      case _:
        msg = "Tried to save an unsupported mode."
        raise ValueError(msg)
    logger.info("Saved %s to %s", self.mode, str(full_name))
    return full_name

  def save_full_model(
    self,
    full_name: Path,
    epoch: int,
  ) -> Path:
    """Save all states.

    Note that here the loss is the loss class.

    """
    torch.save(
      {
        "model_state_dict": self.net.state_dict(),
        "epoch": epoch,
        "loss": self.criterion,
        "optimizer_state_dict": self.optim.state_dict(),
      },
      full_name,
    )
    return full_name

  def save_state_dict(self, full_name: Path) -> Path:
    """Save state_dict of model-only.

    This is useful to load for inference, not for retraining.
    """
    torch.save(self.net.state_dict(), str(full_name))
    return full_name
