"""Print some input data visuals."""

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from torch_practice.dataloading import get_dataloaders
from torch_practice.default_config import default_config
from torch_practice.nn_arch import DynamicAE

if __name__ == "__main__":
  # image-label tuples.
  training_data = torchvision.datasets.CIFAR10("./data", download=True)
  hw_inches = 8  # width, height in inches.
  figure = plt.figure(figsize=(hw_inches, hw_inches))
  cols, rows = 3, 3  # grid shape.
  for i in range(1, cols * rows + 1):  # 1 to 9 inclusive.
    sample_idx = int(torch.randint(len(training_data), size=(1,)).item())
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(str(label))
    plt.axis("off")
    plt.imshow(img)
  plt.show()

  writer = SummaryWriter("runs/torch_practice")

  config = default_config()
  trainloader, _, _ = get_dataloaders(default_config())
  # get some random training images
  dataiter = iter(trainloader)
  images, labels = next(dataiter)

  # create grid of images
  img_grid = torchvision.utils.make_grid(images)

  img_grid = img_grid / 2 + 0.5  # un normalize
  writer.add_image("images", img_grid)

  net = DynamicAE(config)
  net(images)  # needs forward pass
  writer.add_graph(net, images)  # this is very cool..
  writer.close()

  def show_mplib(img_grid: torch.Tensor) -> None:
    """Show imgs using matplotlib."""
    npimg = img_grid.numpy()
    npimg = npimg.transpose((1, 2, 0))
    plt.imshow(npimg)

    plt.show()

else:
  msg = "Use this script with `python -m`, not import."
  raise RuntimeError(msg)
