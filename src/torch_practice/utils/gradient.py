import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from torch_practice.nn_arch import DynamicAE

logger = logging.getLogger(__package__)


# We need to create a nice gradient logger, or DynamicAE Logger.
def get_gradient_statistics(
  net: DynamicAE,
  writer: SummaryWriter,
  epoch: int,
  *,
  full_breakdown: bool = False,
) -> None:
  """Log to console and write gradient analysis to TBoard."""
  my_dict: dict[str, float] = defaultdict(float)
  for name, param in net.named_parameters():
    if param.grad is not None:
      g = param.grad
      avg_g = cpu_result(g.mean())
      my_dict[name] = avg_g
      if full_breakdown:
        min_g = cpu_result(g.min())
        max_g = cpu_result(g.max())
        abs_mean_g = cpu_result(g.abs().mean())
        writer.add_scalars(
          name,
          {"min": min_g, "max": max_g, "mean": avg_g, "abs": abs_mean_g},
          epoch,
        )
      # as "debug" to avoid cluttering the console.

  sort = sorted(my_dict.items(), key=lambda item: abs(item[1]))[0:7]
  logger.debug("5 mean gradients closest to 0: %s\n", sort)
  data = dict_of_lists_to_numpy(my_dict)
  writer.add_scalars("Average Gradient Per Layer", data, epoch)


def cpu_result(p: torch.Tensor) -> float:
  """Tensor to cpu."""
  return p.detach().cpu().item()


def dict_of_lists_to_numpy(
  my_dict: dict[str, float],
) -> dict[str, np.ndarray]:
  """Turn the dict of lists to dict of numpy arrays for TBoard."""
  my_new_dict: dict[str, np.ndarray] = {}
  for k, v in my_dict.items():
    my_new_dict[k] = np.array(v)
  return my_new_dict
