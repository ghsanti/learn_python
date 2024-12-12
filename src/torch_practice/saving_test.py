"""Handle model saving."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from torch_practice.default_config import default_config
from torch_practice.loading import (
  load_full_model,
  load_state_dict,
)
from torch_practice.main_types import RunConfig
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save

if TYPE_CHECKING:
  from torch_practice.main_types import DAEConfig
  from torch_practice.saving import SaveModeType, SaverBaseArgs
else:
  SaverBaseArgs = None
  SaveModeType = None
  DAEConfig = None

logger = logging.getLogger(__package__)


def minimal_model(
  tmp_path: Path,
  model_type: SaveModeType,
) -> tuple[DynamicAE, RunConfig]:
  # configure minimal model
  config = default_config()
  saver_config: SaverBaseArgs = {
    "basedir": tmp_path,
    "save_mode": model_type,
    "save_every": 3,
    "save_at": "all",  # all models every 3 runs.
  }
  config["saver"] = saver_config
  config["arch"]["layers"] = 1
  # instantiate
  model = DynamicAE(config["arch"])
  # initiate
  model(torch.randn((1, *config["arch"]["input_size"])))
  return model, config


class TestSave:
  def test_save_state_dict(self, tmp_path: Path):
    """Test saving a state dict."""
    model_type = "state_dict"
    model, config = minimal_model(tmp_path, model_type)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), config["lr"])
    assert config["saver"] is not None
    saver = Save(config["saver"], model, criterion, optim)
    saved_path = saver.save_model(0, 0.221)
    assert saved_path.exists()

    result = load_state_dict(model, saved_path)
    assert isinstance(result, tuple)

  def test_save_full_model(self, tmp_path: Path):
    """Test saving a small PyTorch model."""
    model_type = "full_model"

    model, config = minimal_model(tmp_path, model_type)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), config["lr"])
    assert config["saver"] is not None
    saver = Save(config["saver"], model, criterion, optim)
    saved_path = saver.save_model(0, 0.221)
    assert saved_path.exists()

    result = load_full_model(filepath=saved_path, weights_only=False)
    assert isinstance(result, dict)

    named_tuple = model.load_state_dict(result["model_state_dict"])
    assert isinstance(named_tuple, tuple)
    optim.load_state_dict(result["optimizer_state_dict"])
    _ = result["loss"]
    epoch = result["epoch"]
    assert epoch == 0
