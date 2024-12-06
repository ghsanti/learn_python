"""Train the AE."""

import logging
import pprint

import torch
from torch.nn import MSELoss
from torch.optim import SGD

from torch_practice.dataloading import get_dataloaders
from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.get_device import get_device
from torch_practice.utils.io import save_model
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def train(
  config: DAEConfig,
) -> None:
  """Train the AutoEncoder.

  Loads CIFAR10 and trains the model.
  """
  logging.basicConfig(level=config.get("log_level"))
  if config.get("seed") is not None:
    torch.manual_seed(config.get("seed"))
  device = get_device()

  net = DynamicAE(config)
  net(torch.randn(1, *config.get("input_size")))  # initialise all layers

  net = net.to(device)  # done after initialising may prevent issues.
  optimizer = SGD(
    params=net.parameters(),  # weights and biases
    lr=config.get("lr"),
    weight_decay=1e-4,
  )
  criterion = MSELoss()

  train, evaluation, _ = get_dataloaders(config)

  logger.info("Network Configuration: ")
  logger.info(pprint.pformat(config))
  logger.debug("Network Device %s", device)
  logger.info("Optimizer %s", optimizer.__class__.__name__)
  logger.info("Loss with %s", criterion.__class__.__name__)

  logger.info("train batches: %s", len(train))
  logger.info("eval batches: %s", len(evaluation))

  train_losses, eval_losses = [], []
  best_eval_loss = (
    float("inf") if config.get("loss_mode") == "min" else float("-inf")
  )
  epochs = config.get("epochs")

  for i in range(epochs):
    train_loss, eval_loss = 0.0, 0.0

    net.train()
    for images, _ in train:
      imgs = images.to(device)
      optimizer.zero_grad()
      loss = criterion(net(imgs), imgs)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    net.eval()
    for images_ev, _ in evaluation:
      imgs_ev = images_ev.to(device)
      with torch.no_grad():
        eval_loss += criterion(net(imgs_ev), imgs_ev).item()

    train_loss = train_loss / len(train)
    train_losses.append(train_loss)

    eval_loss = eval_loss / len(evaluation)
    eval_losses.append(eval_loss)

    if config.get("gradient_log"):
      for name, param in net.named_parameters():
        if param.grad is not None:
          logging.debug("Gradient for %s: %s", name, param.grad.abs().max())

    # print epoch logs
    msg = f"Epoch {i+1} of {epochs}, Train Loss: {train_loss:.3f}"
    logger.info(msg)
    logger.info("eval loss: %s", eval_loss)

    # saving
    name = f"{i}_{eval_loss:.3f}"
    improved = loss_improved(
      best_eval_loss,
      eval_loss,
      config,
    )
    if improved:
      best_eval_loss = eval_loss
    save = config.get("save")
    if save == "all" or improved:
      # save='best' deletes previous "best_" file.
      save_model(net, config, name)


if __name__ == "__main__":
  # for python debugger
  from torch_practice.default_config import default_config

  train(default_config())
