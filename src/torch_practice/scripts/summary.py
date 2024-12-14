"""Print the default network summary."""

if __name__ == "__main__":
  import torch
  from torchinfo import summary

  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  config = default_config()  # you can tweak "config"

  config["arch"] = {
    # architecture
    "growth": 2,
    "init_out_channels": 14,
    "layers": 3,
    "input_size": (3, 32, 32),
    # convolution
    "c_kernel": 2,
    "c_stride": 1,
    "c_activ": torch.nn.functional.leaky_relu,
    # pool
    "use_pool": True,
    "p_kernel": 2,
    "p_stride": 1,
    # dropout
    "use_dropout2d": True,
    "dropout2d_rate": 0.3,
    "dropout_rate_latent": 0.3,
    "use_dropout_latent": False,
    # dense
    "latent_dimension": 72,
    "dense_activ": torch.nn.functional.leaky_relu,
  }
  model = DynamicAE(config["arch"])
  summary(
    model,
    input_size=(1, *config["arch"]["input_size"]),
    device="cpu",
    depth=3,
  )
