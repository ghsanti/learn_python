"""Utilities."""

import logging
import pickle
from pathlib import Path

import torch
from ray.train import get_checkpoint
from torch import nn, optim
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

from .main_types import CIFAR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data(data_dir: str = "./data") -> tuple[CIFAR, CIFAR]:
    """Download CIFAR10 (10 classes) dataset.

    Dataset is not downloaded again if it's available in `data_dir`.
    """
    transform = transforms.Compose(  # list of transformations.
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    )

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_set, test_set


def to_gpu_if_available(net: nn.Module) -> str:
    """Send to the GPU if available."""
    device = "cpu"
    if torch.cuda.is_available():
        logger.info("GPU is available")
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            logger.info("Trying GPU in parallel.")
            net = nn.DataParallel(net)
    net.to(device)
    return device


def split_train(train_set: CIFAR, train_frac: float = 0.8) -> tuple[Subset, Subset]:
    """Split training set into train and validation."""
    new_train_l = int(len(train_set) * train_frac)
    val_l = len(train_set) - new_train_l
    train_subset, val_subset = random_split(
        train_set,
        [new_train_l, val_l],
    )
    return train_subset, val_subset


def reload_state(net: nn.Module, optimizer: optim.Optimizer) -> int:
    """Reload the Network state from checkpoint."""
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with Path.open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    return start_epoch


def test_accuracy(net: nn.Module, device: str = "cpu") -> float:
    """Test accuracy."""
    _, test_set = load_data()

    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
