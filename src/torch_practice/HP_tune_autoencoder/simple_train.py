"""Train the AE."""

import time

import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from torch_practice.dataloading import load_data
from torch_practice.HP_tune_autoencoder.main_types import (
  DAEConfig,
  default_config,
)
from torch_practice.HP_tune_autoencoder.nn_arch import DynamicAE
from torch_practice.utils import to_device_available


def train(epochs: int, config: DAEConfig) -> None:
  """Train the AutoEncoder.

  Loads CIFAR10 and trains the model.
  """
  net = DynamicAE(config)
  net(torch.rand(1, 3, 32, 32))  # initalise decoder.
  net, device = to_device_available(net)
  optimizer = SGD(
    params=net.parameters(),
    lr=config.get("lr"),
    weight_decay=1e-4,
  )
  criterion = MSELoss()

  train, _ = load_data()
  train_set, eval_set = random_split(train, [0.8, 0.2])
  train = DataLoader(
    batch_size=config.get("batch_size"),
    dataset=train_set,
    shuffle=True,
    num_workers=2,
  )
  evaluation = DataLoader(
    batch_size=config.get("batch_size"),
    dataset=eval_set,
    shuffle=True,
    num_workers=2,
  )

  train_loss = []
  for i in range(epochs):
    running_loss = 0.0
    eval_loss = 0.0

    for imgs, _ in train:
      optimizer.zero_grad()
      images = imgs.to(device)
      r = net(images)
      loss = criterion(r, images)
      loss.backward()
      torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
      torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
      optimizer.step()
      running_loss += loss.item()
    for imgs_ev, _ in evaluation:
      with torch.no_grad():
        images_ev = imgs_ev.to(device)
        out = net(images_ev)
        eval_loss += criterion(out, images_ev)

    # check_gradient_statistics(net.named_parameters())

    loss = running_loss / len(train)
    train_loss.append(loss)
    print(f"Epoch {i+1} of {epochs}, Train Loss: {loss:.3f}")

    eval_loss = eval_loss / len(evaluation)
    print("eval loss: ", eval_loss)
    # for name, param in net.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name}: {param.grad.abs().max()}")


if __name__ == "__main__":
  s = time.time()
  config = default_config()
  train(epochs=18, config=config)
  e = time.time()
  print(config.get("batch_size"), " : ", e - s)
