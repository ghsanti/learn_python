"""Handle model saving."""

import logging
from pathlib import Path

import torch

from torch_practice.default_config import default_config
from torch_practice.loading import Loader
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import Save, SaveModeType

logger = logging.getLogger(__package__)


def minimal_model(
  tmp_path: Path,
  model_type: SaveModeType,
) -> tuple[DynamicAE, Save]:
  # configure minimal model
  saver = Save(basedir=tmp_path, save_mode=model_type)
  config = default_config(saver)
  config["layers"] = 1
  # instantiate
  model = DynamicAE(config)
  # initiate
  model(torch.randn((1, *config.get("input_size"))))
  return model, saver


class TestSave:
  def test_save_state_dict(self, tmp_path: Path):
    """Test saving a small PyTorch model."""
    model_type = "inference"
    model, saver = minimal_model(tmp_path, model_type)

    saved_path = saver.save_inference(model, 0, 0.221)
    assert saved_path.exists()

    loader = Loader(model_type)
    loader.from_filename(saved_path, model)

  def test_save_checkpoint(self, tmp_path: Path):
    """Test saving a small PyTorch model."""
    model_type = "training"
    model, saver = minimal_model(tmp_path, model_type)
    optimizer = torch.optim.SGD(model.parameters())
    criterion = torch.nn.MSELoss()

    saved_path = saver.save_checkpoint(model, 0, criterion, 0.221, optimizer)
    assert saved_path.exists()

    loader = Loader(model_type)
    loader.from_filename(path_to_model=saved_path, net=None)
