"""Show how the trained network reconstructs images."""

import logging
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

from torch_practice.default_config import default_config
from torch_practice.loading import (
  get_best_path,
  load_full_model,
  load_state_dict,
)
from torch_practice.main_types import RunConfig, SaveModeType
from torch_practice.nn_arch import DynamicAE


def reconstruct_tboard(
  net: DynamicAE,
  save_mode: SaveModeType,
  config: RunConfig,
  recurse_depth: int = 1,
) -> None:
  """Save a batch of predictions and originals to open in TensorBoard."""
  start_from = Path(config["saver"]["basedir"])
  loss_mode = c["loss_mode"]
  arch = config["arch"]
  depth = recurse_depth

  net(torch.randn(1, *arch["input_size"]))
  best_path = get_best_path(start_from, loss_mode, depth, save_mode)
  if best_path is not None:
    path, _ = best_path
    if save_mode == "state_dict":
      load_state_dict(net, path)
      net.eval()
    else:
      ckp = load_full_model(best_path[0], weights_only=False)
      net.load_state_dict(ckp["model_state_dict"])
      net.eval()

    writer = SummaryWriter("tboard_logs")

    data = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10(
        c["data_dir"],
        train=False,
        transform=transforms.Compose(  # list of transformations.
          [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ],
        ),
      ),
      batch_size=12,
      shuffle=True,
    )
    # create grid of images
    with torch.no_grad():
      imgs, _ = next(iter(data))

      r = net(imgs)
      imgs = imgs / 2 + 0.5
      r = r / 2 + 0.5
      img_grid = torchvision.utils.make_grid(imgs)
      net_img_grid = torchvision.utils.make_grid(r)  # needs forward pass
      writer.add_image("images1", img_grid)
      writer.add_image("images2", net_img_grid)
    writer.close()


if __name__ == "__main__":
  logging.basicConfig(level="DEBUG")
  c = default_config()
  c["arch"] = {
    # architecture
    "growth": 1.7,
    "init_out_channels": 6,
    "layers": 3,
    "input_size": (3, 32, 32),
    # convolution
    "c_kernel": 2,
    "c_stride": 1,
    "c_activ": torch.nn.functional.leaky_relu,
    # pool
    "use_pool": False,
    "p_kernel": 2,
    "p_stride": 2,
    # dropout
    "use_dropout2d": True,
    "dropout2d_rate": 0.3,
    "dropout_rate_latent": 0.3,
    "use_dropout_latent": False,
    # dense
    "latent_dimension": 96,
    "dense_activ": torch.nn.functional.silu,
  }
  net = DynamicAE(c["arch"])
  reconstruct_tboard(net, "state_dict", c, 1)
