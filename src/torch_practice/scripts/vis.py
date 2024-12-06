"""Print some input data visuals."""

import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from torch_practice.default_config import default_config
from torch_practice.nn_arch import DynamicAE

if __name__ == "__main__":
  # image-label tuples.

  writer = SummaryWriter("runs/torch_practice")

  config = default_config()
  data = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10("data/", train=False),
  )
  # get some random training images
  dataiter = iter(data)
  images, labels = next(dataiter)

  # create grid of images
  img_grid = torchvision.utils.make_grid(images)

  img_grid = img_grid / 2 + 0.5  # un normalize
  writer.add_image("images", img_grid)

  net = DynamicAE(config)
  net(images)  # needs forward pass
  writer.add_graph(net, images)  # this is very cool..
  writer.close()


else:
  msg = "Use this script with `python -m`, not import."
  raise RuntimeError(msg)
