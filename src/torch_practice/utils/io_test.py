"""Test io operations using tmp_path."""

import torch

from torch_practice.default_config import default_config
from torch_practice.nn_arch import DynamicAE
from torch_practice.utils.io import (
  get_best_path,
  load_model,
  make_savedir,
)


def test_get_best_path(tmp_path):
  """Test if a .pth file is removed.

  Explicitly make a temporary directory rather than using
  `tmp_path` fixture.
  """
  losses = [0.5, 0.6, 0.3, 0.01, 0.55]
  for epoch, loss in enumerate(losses):
    # create "models" in the path.
    file = tmp_path / f"{epoch}_{loss}.pth"
    file.write_bytes(b"hello world")

  # make sure we use the temporary path
  dirname = tmp_path.resolve()

  # test min
  path = get_best_path(dirname, "min")
  assert path.stem.split("_")[-1] == "0.01"

  # test max
  path = get_best_path(dirname, "max")
  assert path.stem.split("_")[-1] == "0.6"


def test_save_load(tmp_path):
  config = default_config()
  config["layers"] = 1
  model = DynamicAE(config)
  model(torch.randn((1, *config.get("input_size"))))
  filename = "best.pth"
  filepath = tmp_path / filename
  torch.save(model.state_dict(), filepath)
  assert filepath.exists()
  loaded = load_model(model, tmp_path, filename)
  assert isinstance(loaded, tuple)


def test_make_savedir(tmp_path):
  dirname = make_savedir(tmp_path)
  assert "202" in dirname.name
