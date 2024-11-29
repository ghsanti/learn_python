"""Utilities."""

import logging
import pickle
from pathlib import Path

import torch
from ray.train import get_checkpoint
from torch import nn, optim

from torch_practice.dataloading import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def to_device_available(net: nn.Module) -> tuple[nn.Module, str]:
    """Send to device (GPU/MPS) if available."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            logger.info("Trying GPU in parallel.")
            net = nn.DataParallel(net)
    elif torch.mps.is_available():
        device = "mps"
    msg = f"Sending network to {device}"
    logger.info(msg)
    net = net.to(device)
    return net, device


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
