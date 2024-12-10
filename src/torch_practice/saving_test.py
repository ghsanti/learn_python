"""Handle model saving."""

import logging
from pathlib import Path

import torch

from torch_practice.default_config import default_config
from torch_practice.loading import load_model
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

    result = load_model(saved_path, weights_only=True)
    assert isinstance(result, dict)
    assert isinstance(model.load_state_dict(result), tuple)

  def test_save_checkpoint(self, tmp_path: Path):
    """Test saving a small PyTorch model."""
    model_type = "training"
    model, saver = minimal_model(tmp_path, model_type)
    optimizer = torch.optim.SGD(model.parameters())
    criterion = torch.nn.MSELoss()

    saved_path = saver.save_checkpoint(
      net=model,
      epoch=0,
      loss=criterion,
      loss_value=0.221,
      optimizer=optimizer,
    )
    assert saved_path.exists()

    result = load_model(filepath=saved_path, weights_only=False)
    assert isinstance(result, dict)

    named_tuple = model.load_state_dict(result["model_state_dict"])
    assert isinstance(named_tuple, tuple)
    optimizer.load_state_dict(result["optimizer_state_dict"])
    loss = result["loss"]
    epoch = result["epoch"]
    assert epoch == 0
