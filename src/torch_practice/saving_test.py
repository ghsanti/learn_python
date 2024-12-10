"""Handle model saving."""

import logging

import pytest
import torch

from torch_practice.default_config import default_config
from torch_practice.loading import Loader
from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save

logger = logging.getLogger(__package__)

LOSS_PATTERN = r".*_(\d+\.\d+)\.pth?$"


@pytest.fixture
def minimal_model(tmp_path):
  # configure minimal model
  config = default_config(Save(basedir=tmp_path))
  config["layers"] = 1
  # instantiate
  model = DynamicAE(config)
  # initiate
  model(torch.randn((1, *config.get("input_size"))))
  return model, config


def test_save_load(tmp_path, minimal_model: tuple[DynamicAE, DAEConfig]):
  model, config = minimal_model

  loader = Loader(model_mode="inference")
  filepath = tmp_path / "best_0.221.pth"
  # save
  torch.save(model.state_dict(), filepath)
  assert filepath.exists()

  # load from filepath
  loaded = loader.from_filename(filepath, model)
  assert isinstance(loaded, tuple)

  # load from directory + loss mode.
  loaded_2 = loader.from_loss_mode(
    tmp_path,
    config.get("loss_mode"),
    model,
    descend_one=False,
  )
  assert isinstance(loaded_2, tuple)

  saver = Save(tmp_path)
  saver.save_inference(model, 0, 0.221)
