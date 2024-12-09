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
from torch_practice.utils.io import Save
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)


def train(config: DAEConfig) -> None:
  """Train the AutoEncoder.

  Loads CIFAR10 and trains the model.
  """
  logging.basicConfig(level=config["log_level"])
  if config["seed"] is not None:
    torch.manual_seed(config["seed"])

  saver = config["saver"]
  device = get_device()
  net = DynamicAE(config)
  net(torch.randn(1, *config["input_size"]))  # initialise all layers

  net = net.to(device)  # done after initialising may prevent issues.
  optimizer = SGD(
    params=net.parameters(),  # weights and biases
    lr=config["lr"],
    weight_decay=1e-4,
  )
  criterion = MSELoss()

  train, evaluation, _ = get_dataloaders(config)

  # print general logs
  logs(config, optimizer, criterion, device)

  best_eval_loss = None
  epochs = config["epochs"]

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
    eval_loss = eval_loss / len(evaluation)

    if config["gradient_log"]:
      log_gradients(net)

    # print epoch logs
    msg = f"Epoch {i+1} of {epochs}, Train Loss: {train_loss:.3f}"
    logger.info(msg)
    logger.info("eval loss: %s", eval_loss)

    improved = loss_improved(
      best_eval_loss,
      eval_loss,
      config["loss_mode"],
    )

    # saving
    if saver.at is None:
      continue

    need_save = saver.save_time(epoch=i)
    if improved and need_save:
      best_eval_loss = eval_loss

    if need_save and (saver.at == "always" or improved):
      if saver.mode == "inference":
        saver.save_inference(net, i, eval_loss)
      elif saver.mode == "training":
        saver.save_checkpoint(
          net,
          epoch=i,
          loss=criterion,
          loss_value=train_loss,
          optimizer=optimizer,
        )


def logs(
  config: DAEConfig,
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


if __name__ == "__main__":
  # for python debugger
  from torch_practice.default_config import default_config

  # saver default configuration.
  basedir = "checkpoints"
  save_every = 3
  save_for = "inference"
  save_at = "better"
  saver = Save(basedir, save_every, save_for, save_at)

  config = default_config(saver)
  config["layers"] = 1
  config["latent_dimension"] = 12
  train(config)
