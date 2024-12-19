"""Load Cifar10."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .main_types import CIFAR, RunConfig
from .utils import seed_worker


def get_dataloaders(
  config: RunConfig,
) -> tuple[CIFAR, CIFAR, CIFAR]:
  """Get CIFAR10 train,eval,test dataloaders.

  Return:
      train and test dataloaders.

  Dataset is not downloaded again if it's available in `data_dir`.

  """
  transform = transforms.Compose(  # list of transformations.
    [
      transforms.ToTensor(),
    ],
  )

  data_dir = config["data_dir"]
  batch_size = config["batch_size"]
  n_workers = config["n_workers"]
  seed = config["seed"]
  train_set = CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
  )

  test_set = CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
  )

  gen, s_fn = None, None
  if seed is not None:
    gen = torch.Generator()
    gen.manual_seed(0)
    s_fn = seed_worker.seed_worker

  train_subset, eval_subset = random_split(train_set, config["prob_split"])
  train = DataLoader[CIFAR10](
    batch_size=batch_size,
    dataset=train_subset,
    shuffle=True,
    num_workers=n_workers,
    generator=gen,
    worker_init_fn=s_fn,
  )
  evaluation = DataLoader[CIFAR10](
    batch_size=batch_size,
    dataset=eval_subset,
    shuffle=True,
    num_workers=n_workers,
    generator=gen,
    worker_init_fn=s_fn,
  )
  test = DataLoader[CIFAR10](
    batch_size=batch_size,
    dataset=test_set,
    shuffle=True,
    num_workers=n_workers,
    generator=gen,
    worker_init_fn=s_fn,
  )
  return train, evaluation, test
