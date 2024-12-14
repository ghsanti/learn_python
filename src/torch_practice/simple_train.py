"""Train the AE."""

import logging
import pprint

import torch
from torch.nn import MSELoss
from torch.optim import SGD

from torch_practice.dataloading import get_dataloaders
from torch_practice.main_types import RunConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save
from torch_practice.utils.device import get_device_name
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def train(config: RunConfig) -> None:
  """Train the AutoEncoder.

  Loads CIFAR10 and trains the model.
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
  logs(config, optimizer, criterion, device.type)

  best_eval_loss = None
  epochs = config["epochs"]

  for i in range(epochs):
    train_loss, eval_loss = 0.0, 0.0

    net.train()
    for images, _ in train:
      imgs = images.to(device)
      optimizer.zero_grad()
      loss = criterion(net(imgs), imgs)
      loss.backward()  # updates occur per batch.
      optimizer.step()
      train_loss += loss.item()

    net.eval()
    for images_ev, _ in evaluation:
      imgs_ev = images_ev.to(device)
      with torch.no_grad():
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


def logs(
  config: RunConfig,
  optimizer: object,
  criterion: object,
  device: str,
) -> None:
  """Print general logs at the start of optimisation."""
  logger.info("Network Configuration: ")
  logger.info(pprint.pformat(config))
  logger.debug("Network Device %s", device)
  logger.info("Optimizer %s", optimizer.__class__.__name__)
  logger.info("Loss with %s", criterion.__class__.__name__)


def log_gradients(net: DynamicAE) -> None:
  """Log the maximum absolute value of each parameter tensor."""
  for name, param in net.named_parameters():
    if param.grad is not None:
      logging.debug("Gradient for %s: %s", name, param.grad.abs().max())


def epoch_logs(
  index: int,
  epochs: int,
  train_loss: float,
  eval_loss: float,
) -> None:
  """At-epoch-end logger."""
  msg1 = f"Epoch {index+1} / {epochs}"
  msg2 = f"loss train: {train_loss:.3f}"
  msg3 = f"eval: {eval_loss:.3f}"
  msg = f"{msg1}  |  {msg2}  |  {msg3}"
  logger.info(msg)


if __name__ == "__main__":
  # for python debugger
  from torch_practice.default_config import default_config

  config = default_config()
  arch = config["arch"]
  arch["layers"] = 1
  arch["latent_dimension"] = 12
  train(config)
