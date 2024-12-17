"""Train the AE."""

import logging

import torch
from torch.nn import MSELoss
from torch.optim import SGD
from tqdm import tqdm

from torch_practice.dataloading import get_dataloaders
from torch_practice.logging import epoch_logs, log_gradients, logs
from torch_practice.main_types import RunConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save
from torch_practice.utils.device import get_device_name
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def train(config: RunConfig) -> None:
  """Train the AutoEncoder.

  Args:
    config: full "runtime" configuration dictionary.

  This function runs a simple training using the configuration and CIFA10.

  CIFAR10 is loaded (downloaded only if needed).

  The Loss and Optimizer are currently fixed (MSE and SGD respectively.)

  """
  logging.basicConfig(level=config["log_level"])
  if config["seed"] is not None:
    torch.manual_seed(config["seed"])

  device = torch.device(get_device_name())
  net = DynamicAE(config["arch"])
  net(torch.randn(1, *config["arch"]["input_size"]))  # initialise all layers

  net = net.to(device)  # after initialising and before `net.parameters()`
  optimizer = SGD(
    params=net.parameters(),  # weights and biases
    lr=config["lr"],
    weight_decay=1e-4,
  )
  criterion = MSELoss()
  saver = (
    Save(config["saver"], net, criterion, optimizer)
    if config["saver"] is not None
    else None
  )
  train, evaluation, _ = get_dataloaders(config)

  # print general logs
  logs(net, config, optimizer, criterion, device.type)

  best_eval_loss = None
  epochs = config["epochs"]

  for i in range(epochs):
    train_loss, eval_loss = 0.0, 0.0

    net.train()
    for images, _ in tqdm(train):
      imgs = images.to(device)
      optimizer.zero_grad()
      with torch.autocast(
        device_type=device.type,
        dtype=config["autocast_dtype"],
        enabled=config["autocast_dtype"] is not None,
      ):
        loss = criterion(net(imgs), imgs)

      loss.backward()  # updates occur per batch.
      optimizer.step()
      train_loss += loss.item()

    net.eval()
    for images_ev, _ in tqdm(evaluation):
      imgs_ev = images_ev.to(device)
      with (
        torch.no_grad(),
        torch.autocast(
          device_type=device.type,
          dtype=config["autocast_dtype"],
          enabled=config["autocast_dtype"] is not None,
        ),
      ):
        eval_loss += criterion(net(imgs_ev), imgs_ev).item()

    train_loss = train_loss / len(train)
    eval_loss = eval_loss / len(evaluation)

    if config["gradient_log"]:
      log_gradients(net)

    # print epoch logs
    epoch_logs(i, epochs, train_loss, eval_loss)

    # saving
    if saver is not None and saver.save_time(epoch=i):
      improved = loss_improved(
        best_eval_loss,
        eval_loss,
        config["loss_mode"],
      )
      if improved:
        best_eval_loss = eval_loss

      if saver.at == "all" or improved:
        saver.save_model(i + 1, eval_loss)


if __name__ == "__main__":
  # for python debugger
  from torch_practice.default_config import default_config

  c = default_config()
  c["epochs"] = 400
  c["batch_size"] = 12
  c["autocast_dtype"] = None  # torch.bfloat16
  c["saver"]["save_every"] = 10
  c["arch"]["c_activ"] = torch.nn.functional.relu
  c["arch"]["dense_activ"] = torch.nn.functional.silu
  c["arch"]["growth"] = 1.7
  c["arch"]["layers"] = 3
  c["arch"]["c_stride"] = 1
  train(c)
