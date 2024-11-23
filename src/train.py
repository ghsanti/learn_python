"""test hypertuning torch model."""

import logging
import tempfile
from functools import partial
from pathlib import Path

import ray.cloudpickle as pickle
import torch
import torch.nn.functional as F
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch import nn, optim

from .main_types import HPConfig
from .utils import (
    load_data,
    reload_state,
    split_train,
    test_accuracy,
    to_gpu_if_available,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Net(nn.Module):
    """Define a network."""

    def __init__(self, l1: int = 120, l2: int = 84) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # when flattened
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_cifar(config: HPConfig, data_dir: str | None = None) -> None:
    """Train CIFAR10 model."""
    net = Net(config["l1"], config["l2"])
    device = to_gpu_if_available(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=config["lr"], momentum=0.9)

    # if there checkpoint, restores optim state and nn weights.
    start_epoch = reload_state(net, optimizer)

    # load CIFAR10, make train, val, test.
    train_set, _ = load_data(data_dir) if data_dir else load_data()
    train_subset, val_subset = split_train(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):  # i-th batch.
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            limit = 1999
            if i % 2000 == limit:  # print every 2000 mini-batches
                msg = f"[{epoch + 1:d}, {i + 1:5d}] loss: {running_loss / limit+1:.3f}"
                logger.info(
                    msg,
                )
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
            val_steps += 1

        # after 1 training iteration and evaluation,
        # we save (overwrite?) the state.
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with Path.open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    logger.info("Finished Training")


def run_all(
    num_samples: int = 10,
    max_num_epochs: int = 10,
    gpus_per_trial: int = 2,
) -> None:
    """Run the training."""
    data_dir = Path("./data").absolute()
    load_data(str(data_dir))
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=str(data_dir)),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    if best_trial is None:
        msg = "Could not find best trial."
        raise RuntimeError(msg)

    if best_trial.config is not None:
        best_trial_config = f"Best trial config: {best_trial.config}"
        logger.info(best_trial_config)
        best_trial_val_loss = (
            f"Best trial final validation loss: {best_trial.last_result['loss']}"
        )
        logger.info(best_trial_val_loss)
        best_trial_val_acc = f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}"
        logger.info(
            best_trial_val_acc,
        )

        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint = result.get_best_checkpoint(
            trial=best_trial,
            metric="accuracy",
            mode="max",
        )
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with Path.open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
            test_acc = test_accuracy(best_trained_model, device)
            best = f"Best trial test set accuracy: {test_acc}"
            logger.info(best)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    run_all(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
