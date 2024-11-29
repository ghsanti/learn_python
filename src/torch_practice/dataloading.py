from torchvision import datasets, transforms

from torch_practice.HP_tune_classifier.main_types import CIFAR


def load_data(data_dir: str = "./data") -> tuple[CIFAR, CIFAR]:
    """Download CIFAR10 (10 classes) dataset.

    Return:
        train and test dataloaders.

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
