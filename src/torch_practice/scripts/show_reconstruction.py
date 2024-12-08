"""Show how the trained network reconstructs images."""

import logging
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

from torch_practice.default_config import default_config
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.io import load_model_from_mode

logging.basicConfig(level="DEBUG")

c = default_config()
net = DynamicAE(c)
net(torch.randn(1, *c.get("input_size")))

model_dir = Path("checkpoints") / "2024_12_08_T15_17_16Z"
load_model_from_mode(net, model_dir, c.get("loss_mode"))
net.eval()

writer = SummaryWriter("tboard_logs")

data = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(
    "data/",
    train=False,
    transform=transforms.Compose(  # list of transformations.
      [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ],
    ),
  ),
  batch_size=12,
)
# create grid of images
imgs, labels = next(iter(data))
img_grid = torchvision.utils.make_grid(imgs)

img_grid = img_grid / 2 + 0.5  # un normalize
writer.add_image("images1", img_grid)

net_img_grid = torchvision.utils.make_grid(net(imgs))  # needs forward pass
net_img_grid = net_img_grid / 2 + 0.5  # un normalize
writer.add_image("images2", net_img_grid)
writer.close()
