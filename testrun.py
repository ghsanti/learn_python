"""Little test script, since notebooks seem to leak memory on Mac."""

if __name__ == "__main__":
  import logging

  import torch

  from torch_practice.default_config import default_config
  from torch_practice.simple_train import train

  logging.basicConfig(level="DEBUG")
  c = default_config()
  c["epochs"] = 400
  c["batch_size"] = 12
  c["n_workers"] = 0
  c["saver"]["save_every"] = 10
  c["arch"]["c_activ"] = torch.nn.functional.relu
  c["arch"]["dense_activ"] = torch.nn.functional.relu
  c["arch"]["growth"] = 1.7
  c["arch"]["layers"] = 1
  c["arch"]["c_stride"] = 1

  train(c)
