from ray import tune
from ray.tune.schedulers import ASHAScheduler

from torch_practice.trainer import trainer


def main(num_samples: int, max_num_epochs: int, gpus_per_trial: int):
  # HP space.
  config = {
    "c_kernel": tune.choice([2, 3, 4]),  # type: ignore for Ray.
    "c_stride": tune.choice([1, 2]),
    "growth": tune.uniform(1, 4),
    "in_channels": 3,
    "init_out_channels": tune.choice([2, 4, 6, 8]),
    "layers": tune.choice([2, 3, 4, 5]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": 64,
    "use_pool": False,  # tune.choice([False, True]),
    "p_kernel": 2,
    "p_stride": 2,
  }

  # HP Scheduler.
  scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2,
  )
  # run trials.
  result = tune.run(
    trainer,
    resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,  # n of trials / comb of HPs.
    scheduler=scheduler,
  )
  # log statistics
  best_trial = result.get_best_trial("loss", "min", "last")
  print(f"Best trial config: {best_trial.config}")
  print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
  print(
    f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}",
  )

  # best_trained_model = DynamicAE(best_trial.config)
  # device = to_device_available(best_trained_model)
  # # get best trial checkpoint
  # best_checkpoint = result.get_best_checkpoint(
  #     trial=best_trial,
  #     metric="loss",
  #     mode="min",
  # )
  # if isinstance(best_checkpoint, Checkpoint):
  #     with best_checkpoint.as_directory() as checkpoint_dir:
  #         data_path = Path(checkpoint_dir) / "data.pkl"
  #         with data_path.open("rb") as fp:
  #             best_checkpoint_data = pickle.load(fp)

  #         best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
  # test_acc = test_accuracy(best_trained_model, device)
  # print(f"Best trial test set accuracy: {test_acc}")


# def test_accuracy(net, device="cpu"):
#     trainset, testset = load_data()

#     testloader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=2,
#     )
#     criterion = MSELoss()
#     with torch.no_grad():
#         for data in testloader:
#             images = data[0].to(device)
#             outputs = net(images)
#             error = criterion(images, outputs)
#             print(error)


if __name__ == "__main__":
  # You can change the number of GPUs per trial here:
  main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
