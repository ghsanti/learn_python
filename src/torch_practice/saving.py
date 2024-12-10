"""Handle model saving."""

import logging
from pathlib import Path
from typing import Literal

import torch

from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.date_format import assert_date_format, make_timestamp

logger = logging.getLogger(__package__)


SaveAtType = Literal["all", "improve"] | None
# in both cases, it's standard state dict saved, not torchscript
# you can load and convert it if necessary.
SaveModeType = Literal["inference", "training"] | None


class Save:
  def __init__(
    self,
    basedir: str | Path,
    save_every: int = 3,
    save_mode: SaveModeType = "inference",
    save_at: SaveAtType = "improve",
  ) -> None:
    """Handle model and checkpoint saving.

      Upon instantiation, it creates the timestamped directory within basedir.

    Args:
        basedir: path to base directory. Time-stamped subdir is created within.
        save_every: saves every this number of epochs. Must be >0.
        save_mode: save for "training" or "inference".
        save_at: under which circumstances to save.

    """
    if not isinstance(save_every, int) or save_every < 1:
      msg = f"'save_every' must be a natural number. Found {type(save_every)}"
      raise ValueError(msg)
    if save_at not in ["all", "improve", None]:
      msg = f"Options: 'all', 'improve' and None. Received {save_at}."
      raise ValueError(msg)
    self.mode = save_mode
    self.every = save_every
    self.at = save_at
    self.dirname = self._make_savedir(basedir)
    self.user_saving = self.at is not None and self.mode is not None

  # basic utilities for homogeneity.
  def save_time(self, epoch: int) -> bool:
    """Check if it is time to save according to configuration."""
    return ((epoch + 1) % self.every) == 0

  def make_filepath(self, epoch: int, loss: float) -> Path:
    """Make filepath using epoch and loss value."""
    return self.dirname / f"{epoch}_{loss:.3f}.pth"

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

  def save_inference(self, net: DynamicAE, epoch: int, loss: float) -> None:
    """Save state_dict of model-only.

    This is useful to load for inference, not for retraining.
    """
    full_name = self.make_filepath(epoch, loss)
    torch.save(net.state_dict(), str(full_name))
    logger.info("Saved model state_dict to %s", str(full_name))

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
