import tempfile
from pathlib import Path

import ray.cloudpickle as pickle
import torch
from ray.train import Checkpoint, get_checkpoint
from torch.nn import MSELoss
from torch.optim import Adam

from .dataloading import get_dataloaders
from .main_types import DAEConfig
from .nn_arch import DynamicAE
from .utils.to_device_available import to_device_available


def trainer(config: DAEConfig) -> None:
  # set up
  nn = DynamicAE(config)
  optimizer = Adam(params=nn.parameters())
  criterion = MSELoss()

  # reload checkpoints if needed.
  checkpoint = get_checkpoint()
  if checkpoint:
    with checkpoint.as_directory() as checkpoint_dir:
      data_path = Path(checkpoint_dir) / "data.pkl"
      # has epoch, net and opt state dicts stored.
      with data_path.open("rb") as fp:
        checkpoint_state = pickle.load(fp)
        start_epoch = checkpoint_state["epoch"]
        nn.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
  else:
    start_epoch = 0

  # device selection
  device = to_device_available(nn)

  # data
  trainloader, valloader, test = get_dataloaders(config)
  # training loop
  for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(trainloader, 0):
      inputs = data[0].to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = nn(inputs)
      loss = criterion(outputs, inputs)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      epoch_steps = i
      if i % 2000 == 1999:  # print every 2000 mini-batches
        print(
          "[%d, %5d] loss: %.3f"
          % (epoch + 1, i + 1, running_loss / epoch_steps),
        )
        running_loss = 0.0

    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for data in valloader:
      with torch.no_grad():
        inputs = data[0].to(device)

        outputs = nn(inputs)
        loss = criterion(outputs, inputs)
        print(loss)

        # val_loss += loss.cpu().numpy()
        # val_steps += 1

    checkpoint_data = {
      "epoch": epoch,
      "net_state_dict": nn.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
      # first dumps the data
      data_path = Path(checkpoint_dir) / "data.pkl"
      with data_path.open("wb") as fp:
        pickle.dump(checkpoint_data, fp)

      # then reloads the checkpoint from the tmp dir.
      checkpoint = Checkpoint.from_directory(checkpoint_dir)
      train.report(
        {"loss": val_loss / val_steps, "accuracy": correct / total},
        checkpoint=checkpoint,
      )
