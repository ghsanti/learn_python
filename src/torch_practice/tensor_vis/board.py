"""Write graph to board."""

if __name__ == "__main":
  import torch
  import torchvision
  from matplotlib import pyplot as plt
  from torch.utils.tensorboard.writer import SummaryWriter

  from torch_practice.dataloading import get_dataloaders
  from torch_practice.default_config import default_config
  from torch_practice.nn_arch.nn_arch import DynamicAE

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
