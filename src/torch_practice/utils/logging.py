"""Utilities for model, config and other logging.

This module is basically a class RuntimeLogger that helps to both
log to console on user-defined level and to tensorboard, and inherits from
the Logger in stdlib.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchvision
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from .date_format import make_timestamp
from .gradient import get_gradient_statistics
from .pp_dict import pp_dict

if TYPE_CHECKING:
  from torch_practice.main_types import RunConfig
  from torch_practice.nn_arch import DynamicAE
else:
  RunConfig = None
  DynamicAE = None


class RuntimeLogger(logging.Logger):
  def __init__(self, config: RunConfig) -> None:
    """Configure Runtime logging system.

    Args:
      config: runtime configuration.

    """
    base_args = config["logger"]
    self.log_level = base_args["log_level"]
    self.gradients = base_args["gradients"]
    self.network_graph = base_args["network_graph"]
    self.tboard_dir = base_args["tboard_dir"]
    self.timestamp = make_timestamp()
    self.writer = self._set_up_writer()
    self.img_grid_done = False
    self.logger = logging.getLogger(__package__)
    # user can use `logging.basicConfig(...)` for full app logging control.
    self.logger.setLevel(level=self.log_level)

  def general(
    self,
    net: DynamicAE,
    config: RunConfig,
    optimizer: object,
    criterion: object,
    device: str,
  ) -> None:
    """Print general logs at the start of optimisation."""
    self.logger.info("Torch Version: %s", torch.__version__)
    self.logger.info(
      "__Runtime Configuration__\n\n%s\n",
      pp_dict(config, 4),
    )
    # optional network summary.
    if self.network_graph is True:
      from torchinfo import summary

      result_stats = summary(
        net,
        (1, *config["arch"]["input_size"]),
        device=device,
        verbose=0,
      )
      self.logger.info(str(result_stats))
    self.logger.debug("Network Device %s", device)
    self.logger.info("Optimizer %s", optimizer.__class__.__name__)
    self.logger.info("Loss with %s", criterion.__class__.__name__)

  def tboard_gradient_stats(
    self,
    net: DynamicAE,
    epoch: int,
  ) -> None:
    """Log debug summary to console and write gradient analysis to TBoard."""
    if self.gradients and self.writer:
      get_gradient_statistics(net, self.writer, epoch)

  def tboard_inference_on_batch(
    self,
    net: DynamicAE,
    device: str,
    test_batch: torch.Tensor,
    epoch: int = 0,
  ) -> None:
    """Write inference of a set of images to tensorboard."""
    if self.writer is not None:
      with torch.no_grad():
        net.eval()
        net = net.to(device)
        imgs = test_batch.to(device)
        r = net(imgs)
        r = r / 2 + 0.5
        if not self.img_grid_done:
          imgs = imgs / 2 + 0.5
          img_grid = torchvision.utils.make_grid(imgs)
          self.writer.add_image("images_original", img_grid, epoch)
          self.img_grid_done = True
        net_img_grid = torchvision.utils.make_grid(r)  # needs forward pass
        # no index needed, see tboard magic
        self.writer.add_image("images_predicted", net_img_grid, epoch)

  def last_lr(self, lr_scheduler: LRScheduler) -> None:
    """Log last learning rate used by scheduler."""
    try:
      self.logger.debug("Current learning rate %s", lr_scheduler.get_last_lr())
    except NotImplementedError:
      self.logger.warning("Tried to get current learning rate but failed.")

  def on_epoch_end(
    self,
    index: int,
    epochs: int,
    train_loss: float,
    eval_loss: float,
  ) -> None:
    """At-epoch-end actions."""
    if self.writer:
      self.writer.add_scalars(
        "Avg Batch Loss",
        {"train": train_loss, "eval": eval_loss},
        index + 1,
      )
      self.writer.flush()

    msg1 = f"Epoch {index+1} / {epochs}"
    msg2 = f"loss train: {train_loss:.3f}"
    msg3 = f"eval: {eval_loss:.3f}"
    msg = f"{msg1}  |  {msg2}  |  {msg3}"
    self.logger.info(msg)

  def _set_up_writer(self) -> SummaryWriter | None:
    """If tboard_dir is set, it creates a writer."""
    if self.tboard_dir is not None:
      subdir = f"{__package__}_{self.timestamp}"
      writer = SummaryWriter(Path(self.tboard_dir) / subdir)
    else:
      writer = None
    return writer
