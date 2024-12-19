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
  """Retrieve best model and save batch of predictions and originals."""
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
    else:
      ckp = load_full_model(best_path[0], weights_only=False)
      net.load_state_dict(ckp["model_state_dict"])

    net.eval()
    writer = SummaryWriter("runs")  # c["tboard_dir"])

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
      batch_size=16,
    )
    # create grid of images

    with torch.no_grad():
      imgs, _ = next(iter(data))

      net.eval()
      r = net(imgs)
      # r = r / 2 + 0.5
      # imgs = imgs / 2 + 0.5
      img_grid = torchvision.utils.make_grid(imgs)
      net_img_grid = torchvision.utils.make_grid(r)  # needs forward pass
      writer.add_image("images_original", img_grid)
      writer.add_image("images2_predicted", net_img_grid)
    writer.close()


if __name__ == "__main__":
  logging.setLevel(level="DEBUG")
  c = default_config()
  c["epochs"] = 400
  c["batch_size"] = 64
  c["autocast_dtype"] = None  # None|torch.bfloat16|torch.float16
  c["saver"]["save_every"] = 10
  c["arch"]["c_activ"] = torch.nn.functional.silu
  c["arch"]["dense_activ"] = torch.nn.functional.silu
  c["arch"]["growth"] = 1.7
  c["arch"]["layers"] = 3
  c["arch"]["c_stride"] = 1
  c["gradient_log"] = True
  c["lr"] = 0.01
  c["arch"]["dropout_rate_latent"] = 0.1
  c["arch"]["dropout2d_rate"] = 0.1
  net = DynamicAE(c["arch"])
  reconstruct_tboard(net, "state_dict", c, 1)
