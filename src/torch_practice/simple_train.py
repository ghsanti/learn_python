"""Train the AE."""

import logging
import time

import torch
from torch.nn import MSELoss
from torch.optim import SGD

from torch_practice.dataloading import get_dataloaders
from torch_practice.default_config import default_config
from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch.nn_arch import DynamicAE
from torch_practice.utils.to_device_available import to_device_available

logger = logging.getLogger(__package__)


def train(
  config: DAEConfig,
) -> None:
  """Train the AutoEncoder.

  Loads CIFAR10 and trains the model.
  """
  logging.basicConfig(level=config.get("log_level"))
  net = DynamicAE(config)
  device = to_device_available(net)
  optimizer = SGD(
    params=net.parameters(),
    lr=config.get("lr"),
    weight_decay=1e-4,
  )
  criterion = MSELoss()

  # logging.basicConfig(logging.WARN) skips info/debug msgs.
  logger.info("Network Configuration %s", config)
  logger.debug("Network Device %s", device)
  logger.info("Optimizer %s", optimizer.__class__.__name__)
  logger.info("Loss with %s", criterion.__class__.__name__)

  train, evaluation, test = get_dataloaders(config)

  logger.info("train batches: %s", len(train))
  logger.info("eval batches: %s", len(evaluation))

  train_loss = []
  epochs = config.get("epochs")
  for i in range(epochs):
    running_loss = 0.0
    eval_loss = 0.0

    for imgs, _ in train:
      optimizer.zero_grad()
      images = imgs.to(device)
      r = net(images)
      loss = criterion(r, images)
      loss.backward()
      if config.get("clip_gradient_value"):
        torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
      if config.get("clip_gradient_norm"):
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
      optimizer.step()
      running_loss += loss.item()
    for imgs_ev, _ in evaluation:
      with torch.no_grad():
        images_ev = imgs_ev.to(device)
        out = net(images_ev)
        eval_loss += criterion(out, images_ev).item()

    # need to check how to log this stuff correctly
    # check_gradient_statistics(net.named_parameters())

    loss = running_loss / len(train)
    msg = f"Epoch {i+1} of {epochs}, Train Loss: {loss:.3f}"
    logger.info(msg)

    train_loss.append(loss)

    eval_loss = eval_loss / len(evaluation)
    logger.info("eval loss: %s", eval_loss)

    for name, param in net.named_parameters():
      if param.grad is not None:
        logging.debug("Gradient for %s: %s", name, param.grad.abs().max())


if __name__ == "__main__":
  s = time.time()
  config = default_config()
  if config.get("seed") is not None:
    import torch

    torch.manual_seed(config.get("seed"))
  config["layers"] = 3
  config["latent_dimension"] = 28
  train(config=config)
  e = time.time()
  logger.info("%s : %s", config.get("batch_size"), e - s)
