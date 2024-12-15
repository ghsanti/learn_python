"""Simple profiling run with default configuration."""

if __name__ == "__main__":
  import logging

  lgr = logging.getLogger()
  logging.basicConfig(level="DEBUG")  # default is warn
  import torch
  from torchinfo import summary

  # abs imports if can be __main__ (script)
  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  config = default_config()  # you can tweak "config"
  config["batch_size"] = 500
  i_size = config["arch"]["input_size"]
  model = DynamicAE(config["arch"])

  # simple profiling info

  device = "mps"  # get_device_name()

  # NOTE: we test inference in training mode, so we need batch > 1
  model(torch.randn((1, *i_size)))  # initialize

  summary(model, input_size=(1, *config["arch"]["input_size"]))

  lgr.debug("Device %s", device)
  model.to(device)
  img = torch.randn(config["batch_size"], *config["arch"]["input_size"]).to(
    device,
  )
  if device == "mps":
    from torch.mps import profiler

    with profiler.profile(mode="interval", wait_until_completed=False):
      model(img)

  else:
    from torch.profiler import ProfilerActivity, profile, record_function

    model = model.to(device)
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
    lgr.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
