"""Saves model to user defined directory."""

import logging
from pathlib import Path

import torch
from torch import nn

from torch_practice.main_types import DAEConfig


def save_model(net: nn.Module, name: str, config: DAEConfig) -> None:
  """Save the model using metadata from config."""
  dirname = Path(config.get("save_dir")).expanduser()
  dirname.mkdir(parents=True, exist_ok=True)
  logger = logging.getLogger(__package__)
  if config.get("save") == "best_only":
    try:
      file = next(dirname.glob("best_*.pth"))
      logger.info("removing file %s", file)
      file.unlink(missing_ok=True)
    except StopIteration:
      logger.info("No `best_*` file found to remove at %s", dirname)

    name = "best_" + name
  filename = Path(name + ".pth")
  fullname = dirname / filename
  logger.info("Saving %s to %s", config.get("save"), str(fullname))
  torch.save(net.state_dict(), fullname)
