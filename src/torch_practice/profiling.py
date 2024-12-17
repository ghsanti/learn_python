"""Simple profiling run with default configuration."""

import logging
from typing import TYPE_CHECKING

import torch
from torchinfo import summary

from torch_practice.utils.device import get_device_name

logging.basicConfig(level="DEBUG")  # default is warn

logger = logging.getLogger(__package__)


if TYPE_CHECKING:
  from torch_practice.nn_arch import DynamicAE
else:
  DynamicAE = None


def profile_forward(
  model: DynamicAE,
  input_size: tuple[int, int, int],
  batch_size: int,
  device: str | None,
) -> None:
  """Profile an instance of the model (inference).

  The model is still in training mode, so we can check for any weird time
  consumed by BatchNorm, Dropout and such.

  Args:
    model: instance (the instance must be on CPU)
    input_size: tuple of size ints (channels, height, width)
    batch_size: n_samples to pass (use large for more accurate averages.)
    device: where to send it to, or it's automatically chosen (best available.)

  """
  i_size = input_size
  device = device if device is not None else get_device_name()

  # Initialise model (optional layers.)
  model(torch.randn((1, *i_size)))

  summary(model, input_size=(1, *input_size))

  # send to device
  logger.debug("Device %s", device)
  model.to(device)
  img = torch.randn(batch_size, *input_size).to(device)

  if device == "mps":
    from torch.mps import profiler

    with profiler.profile(mode="interval", wait_until_completed=False):
      logger.info("Running Mac MPS profile. Open with XCode Instruments tool.")
      model(img)

  else:
    from torch.profiler import ProfilerActivity, profile, record_function

    acts = [ProfilerActivity.CPU]

    if device == "cuda":
      acts.append(ProfilerActivity.CUDA)
    with (
      profile(
        activities=acts,
        record_shapes=True,
        profile_memory=True,
      ) as prof,
      record_function("model_inference"),
    ):
      model(img)
    logger.info(
      prof.key_averages().table(sort_by="cpu_time_total", row_limit=10),
    )
