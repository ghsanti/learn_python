"""Simple profiling run with default configuration."""

import logging
from typing import TYPE_CHECKING

import torch
from torchinfo import summary

from torch_practice.utils.device import get_device_name

logger = logging.getLogger(__package__)


if TYPE_CHECKING:
  from torch_practice.nn_arch import DynamicAE
else:
  DynamicAE = None


def profile_forward(
  model: DynamicAE,
  input_size: tuple[int, int, int],
  batch_size: int,
  device: str | None = None,
  *,
  print_model: bool = False,
  generate_stack_trace: bool = False,
) -> None:
  """Profile an instance of the model (inference).

  The model is still in training mode, so we can check for any weird time
  consumed by BatchNorm, Dropout and such.

  Args:
    model: instance (the instance must be on CPU)
    input_size: tuple of size ints (channels, height, width)
    batch_size: n_samples to pass (use large for more accurate averages.)
    device: where to send it to, or it's automatically chosen (best available.)
    print_model: whether to print (True) or not (False) a `torchinfo.summary`
    generate_stack_trace: output a `trace.json` only use for cuda or cpu.

  """
  i_size = input_size
  device = device if device is not None else get_device_name()

  # Initialise model (optional layers.)
  model(torch.randn((1, *i_size)))

  if print_model:
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
      if generate_stack_trace:
        logger.info("Stack Trace JSON file is unavailable for MPS.")
        logger.info("Use `generate_stack_trace=False` in that case.")

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
      prof.key_averages().table(
        sort_by=f"self_{device}_time_total",
        row_limit=10,
      ),
    )
    if generate_stack_trace:
      prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  logging.basicConfig(level="DEBUG")  # default is warn
  c = default_config()
  c["epochs"] = 400
  c["autocast_dtype"] = None  # None|torch.bfloat16|torch.float16
  c["saver"]["save_every"] = 10
  c["arch"]["c_activ"] = torch.nn.functional.relu
  c["arch"]["dense_activ"] = torch.nn.functional.silu
  c["arch"]["growth"] = 1.7
  c["arch"]["layers"] = 3
  c["arch"]["c_stride"] = 1
  model = DynamicAE(c["arch"])
  batch_size = 500  # large for +accurate profiling
  profile_forward(model, c["arch"]["input_size"], i, "cpu")
