"""test hypertuning torch model."""

import logging
from functools import partial
from pathlib import Path

import ray.cloudpickle as pickle
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import nn

from .nn_arch import Net
from .trainer import train_cifar
from .utils import (
    test_accuracy,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_all(
    num_samples: int = 10,
    max_num_epochs: int = 10,
    gpus_per_trial: int = 2,
    path_to_data: str = "./data",
) -> None:
    """Run the training."""
    data_dir = Path(path_to_data).absolute()
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
    # train_cifar will store results in ray's dir.
    result = tune.run(
        partial(train_cifar, data_dir=str(data_dir)),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )
    # Trial is holds the HPs and metadata, not the model.
    best_trial = result.get_best_trial("loss", "min", "last")
    if best_trial is None:
        msg = "Could not find best trial."
        raise RuntimeError(msg)
    # Trial metadata used to get Checkpoint (path to best.)
    # in turns this contains our checkpoint_object / state dict.
    best_checkpoint = result.get_best_checkpoint(
        trial=best_trial,
        metric="accuracy",
        mode="max",
    )

    if best_trial.config is not None:
        # here just logs best trial info.
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

        # use the newly found HPs.
        best_trained_model = Net(
            best_trial.config.get("l1"),
            best_trial.config.get("l2"),
        )
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        # send net to device
        best_trained_model.to(device)

        # we load the model with the actual weights now,
        # to run a real test, not just evaluation.
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with data_path.open("rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
            test_acc = test_accuracy(best_trained_model, device)
            best = f"Best trial test set accuracy: {test_acc}"
            logger.info(best)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    run_all(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
