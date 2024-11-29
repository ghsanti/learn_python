"""test hypertuning torch model."""

import logging
import tempfile
from pathlib import Path

import ray.cloudpickle as pickle
import torch
from ray import train
from ray.train import Checkpoint
from torch import nn, optim
from torch.utils.data import random_split

from torch_practice.utils import (
    load_data,
    reload_state,
    to_gpu_if_available,
)

from .main_types import HPConfig
from .nn_arch import Net

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_cifar(config: HPConfig, data_dir: str | None = None) -> None:
    """Train NN on CIFAR10.

    It has a training/eval loop, and stores a checkpoint per epoch.

    Params
        config: HyperParameters
        data_dir: where to store or load dataset to/from.
    """
    net = Net(config.get("l1"), config.get("l2"))
    device = to_gpu_if_available(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=config.get("lr"), momentum=0.9)

    # tries to reload from ray-stored data.
    start_epoch = reload_state(net, optimizer)

    train_set, _ = load_data(data_dir) if data_dir else load_data()
    train_subset, val_subset = random_split(train_set, [0.8, 0.2])

    # make Loaders from Sets
    # they are iterables.
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config.get("batch_size")),
        shuffle=True,  # mix at epoch start.
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config.get("batch_size")),
        shuffle=True,
        num_workers=8,
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):  # i-th batch.
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients (?)
            optimizer.zero_grad()

            # forward + backward + update
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            limit = 1999
            if i % limit:  # print every 2000 mini-batches
                msg = f"[{epoch + 1:d}, {i + 1:5d}] loss: {running_loss / i+1:.3f}"
                logger.info(msg)
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for data in val_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                # ,1 looks at each row (prediction),
                # and gets max column
                # "predicted" are the 'max col' idx or class.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                # here loss needs to be on cpu to call numpy.
                val_loss += loss.cpu().numpy()
            val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # makes a tmp dir in /tmp/adfgsdf in Unix.
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            # tmp dir and contents are removed after
            # upper context exits.
            with data_path.open("wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            # report needs a Checkpoint object.
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    logger.info("Finished Training")
