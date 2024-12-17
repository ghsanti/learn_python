"""Utilities for model, config and other logging."""

import logging
from collections.abc import Callable

import torch

from torch_practice.main_types import RunConfig
from torch_practice.nn_arch import DynamicAE

logger = logging.getLogger(__package__)


def logs(
  net: DynamicAE,
  config: RunConfig,
  optimizer: object,
  criterion: object,
  device: str,
) -> None:
  """Print general logs at the start of optimisation."""
  logger.info("Torch Version: %s", torch.__version__)
  sp_keys = {"arch": "ARCHITECTURE", "saver": "SAVING"}
  logger.info("__Runtime Configuration__\n\n%s\n", str_config(config, sp_keys))
  # optional network summary.
  if config["print_network_graph"] is True:
    from torchinfo import summary

    result_stats = summary(
      net,
      (1, *config["arch"]["input_size"]),
      device=device,
      verbose=0,
    )
    logger.info(str(result_stats))
  logger.debug("Network Device %s", device)
  logger.info("Optimizer %s", optimizer.__class__.__name__)
  logger.info("Loss with %s", criterion.__class__.__name__)


def str_config(
  config: RunConfig | dict,
  log_at: dict[str, str] | None = None,
) -> str:
  """Clean up config classes and names for logging.

  Args:
    config: full configuration object
    log_at: key_name to msg map; logs an added `msg` at specific `key_name`

  """
  to_join = []
  for k, v in config.items():
    if log_at is not None and k in log_at:
      to_join.append("\n" + log_at[k] + "\n")
    new_val = v
    if isinstance(v, Callable):
      new_val = v.__name__
    elif isinstance(v, dict):
      new_val = str_config(v)
    elif not isinstance(v, bool | str | int | float | tuple):
      new_val = v.__class__.__name__
    to_join.append(f"{k} = {new_val}")
  return "\n".join(to_join)


def log_gradients(net: DynamicAE) -> None:
  """Log the maximum absolute value of each parameter tensor."""
  for name, param in net.named_parameters():
    if param.grad is not None:
      logging.debug("Gradient for %s: %s", name, param.grad.abs().max())


def epoch_logs(
  index: int,
  epochs: int,
  train_loss: float,
  eval_loss: float,
) -> None:
  """At-epoch-end logger."""
  msg1 = f"Epoch {index+1} / {epochs}"
  msg2 = f"loss train: {train_loss:.3f}"
  msg3 = f"eval: {eval_loss:.3f}"
  msg = f"{msg1}  |  {msg2}  |  {msg3}"
  logger.info(msg)
