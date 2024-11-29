"""Main application types."""

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torchvision

    class HPConfig(TypedDict):
        """Hyperparameters configuration."""

        l1: int
        l2: int
        lr: float
        batch_size: int

    CIFAR = torchvision.datasets.CIFAR10
else:
    HPConfig = None
    CIFAR = None
