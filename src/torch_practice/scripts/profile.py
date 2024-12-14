"""Simple profiling run with default configuration."""

if __name__ == "__main__":
  import logging

  lgr = logging.getLogger()
  logging.basicConfig(level="DEBUG")  # default is warn
  import torch
  from torch.profiler import ProfilerActivity, profile, record_function
  from torchinfo import summary

  # abs imports if can be __main__ (script)
  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  config = default_config()  # you can tweak "config"
  model = DynamicAE(config["arch"])
  summary(model, input_size=(1, *config["arch"]["input_size"]), device="cpu")

  # simple profiling info

  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.xpu.is_available():
    device = "xpu"
  elif torch.mps.is_available():
    device = "mps"

  lgr.debug("Device %s", device)
  if device == "mps":
    import sys

    lgr.critical("MPS profiling is not available.")
    sys.exit(0)
  else:
    img = torch.randn(config["batch_size"], *config["arch"]["input_size"]).to(
      device,
    )
    model = model.to(device)
    with (
      profile(
        activities=[
          ProfilerActivity.CPU,
          # ProfilerActivity.XPU,
          # ProfilerActivity.CUDA,
          # ProfilerActivity.MPS # may be available in the future.
        ],
        record_shapes=True,
        profile_memory=True,
      ) as prof,
      record_function("model_inference"),
    ):
      # with
      model(img)
    lgr.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
