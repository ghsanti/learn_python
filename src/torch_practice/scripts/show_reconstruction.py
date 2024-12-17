"""Show how the trained network reconstructs images."""

if __name__ == "__main__":
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
  from torch_practice.nn_arch import DynamicAE

  logging.basicConfig(level="DEBUG")

  c = default_config()

  # must the right arch for the weights! *below is for sample weights.*
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
  net(torch.randn(1, *c["arch"]["input_size"]))

  start_from = Path(c["saver"]["basedir"])
  save_mode = "state_dict"  # or "state_dict"
  depth = 1
  loss_mode = c["loss_mode"]
  best_path = get_best_path(start_from, loss_mode, depth, save_mode)
  if best_path is not None:
    path, value = best_path
    if save_mode == "state_dict":
      load_state_dict(net, path)
      net.eval()
    else:
      ckp = load_full_model(best_path[0], weights_only=False)

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
    # writer.add_image("images1", img_grid)
    with torch.no_grad():
      imgs, labels = next(iter(data))

      r = net(imgs)
      print(imgs.shape, r.shape)
      imgs = imgs / 2 + 0.5
      r = r / 2 + 0.5
      img_grid = torchvision.utils.make_grid(imgs)
      net_img_grid = torchvision.utils.make_grid(r)  # needs forward pass
      # print(imgs[0], r[0])
      writer.add_image("images1", img_grid)
      writer.add_image("images2", net_img_grid)
    writer.close()
