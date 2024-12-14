"""Playground for logging parts of the net or optimizer."""

if __name__ == "__main__":
  import logging

  logger = logging.getLogger(__package__)
  logging.basicConfig(level="DEBUG")
  import torch

  from torch_practice.default_config import default_config
  from torch_practice.nn_arch import DynamicAE

  net = DynamicAE(default_config()["arch"])
  optim = torch.optim.SGD(params=net.parameters())

  logger.debug(optim.state_dict)
  logger.debug(
    next(net.named_parameters()),
  )
  rt = torch.randn((1, 3, 32, 32))

  logger.debug(optim.state_dict())
  logger.debug(next(net.named_parameters()))
  net(rt)
