"""Print the default network summary."""

if __name__ == "__main__":
  from torchinfo import summary

  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  config = default_config()  # you can tweak "config"
  model = DynamicAE(config)
  summary(model, input_size=(1, *config.get("input_size")), device="cpu")
else:
  msg = "Use this script with `python -m`, not import."
  raise RuntimeError(msg)
