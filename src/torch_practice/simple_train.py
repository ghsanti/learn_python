"""Train the AE."""

import torch
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch_practice.dataloading import get_dataloaders
from torch_practice.main_types import RunConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save
from torch_practice.utils.device import get_device_name
from torch_practice.utils.logging import RuntimeLogger
from torch_practice.utils.track_loss import loss_improved


def train_ae(config: RunConfig) -> None:
  """Train the AutoEncoder.

  Args:
    config: full "runtime" configuration dictionary.

  This function runs a simple training using the configuration and CIFA10.

  CIFAR10 is loaded (downloaded only if needed).

  The Loss and Optimizer are currently fixed (MSE and SGD respectively.)

  """
  logger = RuntimeLogger(config)
  if config["seed"] is not None:
    torch.manual_seed(config["seed"])

  device = torch.device(get_device_name())
  net = DynamicAE(config["arch"])
  net(torch.randn(1, *config["arch"]["input_size"]))  # initialise all layers

  net = net.to(device)  # after initialising and before `net.parameters()`
  optimizer = SGD(
    params=net.parameters(),
    lr=config["lr"],
    # weight_decay=1e-4,
    momentum=0.9,
    nesterov=True,
  )
  criterion = MSELoss()
  lr_scheduler = ReduceLROnPlateau(
    optimizer,
    patience=config["patience"],
    min_lr=10e-8,
  )
  saver = (
    Save(config["saver"], net, criterion, optimizer)
    if config["saver"] is not None
    else None
  )
  train, evaluation, test = get_dataloaders(config)
  test, _ = next(iter(test))  # use 1 test batch.

  # general logs
  logger.general(net, config, optimizer, criterion, device.type)

  best_eval_loss = None  # we set this when necessary.
  epochs = config["epochs"]

  for i in range(epochs):
    train_loss, eval_loss = 0.0, 0.0

    net.train(mode=True)
    for bi, (images, _) in enumerate(tqdm(train)):
      imgs = images.to(device)
      optimizer.zero_grad()
      with torch.autocast(
        device_type=device.type,
        dtype=config["autocast_dtype"],
        enabled=config["autocast_dtype"] is not None,
      ):
        loss = criterion(net(imgs), imgs)

      loss.backward()  # updates occur per batch.
      clip_grad_norm_(net.parameters(), max_norm=1.0)
      optimizer.step()
      loss_value = loss.item()
      train_loss += loss_value
      tb_x = i * len(train) + bi + 1  # gives us full trace
      if logger.writer is not None:
        logger.writer.add_scalar("Batch Loss", loss_value, tb_x)

    net.train(mode=False)
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
    lr_scheduler.step(eval_loss)
    logger.last_lr(lr_scheduler)

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

    # epoch logs
    logger.tboard_gradient_stats(net, i)
    logger.tboard_inference_on_batch(net, device.type, test, i)
    logger.on_epoch_end(i, epochs, train_loss, eval_loss)


if __name__ == "__main__":
  # for python debugger
  from torch_practice.default_config import default_config

  c = default_config()
  c["epochs"] = 400
  c["batch_size"] = 64
  c["saver"]["save_every"] = 10
  # slower than 32 in local tests.
  c["autocast_dtype"] = None  # None|torch.bfloat16|torch.float16
  c["lr"] = 5e-4
  c["arch"]["init_out_channels"] = 64
  c["arch"]["dense_activ"] = torch.nn.functional.leaky_relu
  c["arch"]["c_activ"] = torch.nn.functional.leaky_relu
  c["arch"]["c_stride"] = 2
  c["arch"]["growth"] = 2
  c["arch"]["layers"] = 3
  c["arch"]["dropout_rate_latent"] = 0.1
  c["arch"]["dropout2d_rate"] = 0.4
  c["arch"]["latent_dimension"] = 72
  train_ae(c)
