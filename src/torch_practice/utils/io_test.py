"""Test io operations using tmp_path."""

from pathlib import Path
from tempfile import TemporaryDirectory

from torch_practice.default_config import default_config
from torch_practice.utils.io import get_best_path, remove_old_best


def test_remove_old_best() -> None:
  """Test if a .pth file is removed.

  Explicitly make a temporary directory rather than using
  `tmp_path` fixture.
  """
  with TemporaryDirectory() as d:
    tmp_path = Path(d)

    file = tmp_path / "best_checkpoint_2.pth"
    file.write_bytes(b"hello world")
    remove_old_best(tmp_path, glob="non_existent_file$")  # no file removed
    assert file.exists()  # should still be there
    remove_old_best(tmp_path)
    assert not file.exists()  # now should be removed


def test_get_best_path() -> None:
  """Test if a .pth file is removed.

  Explicitly make a temporary directory rather than using
  `tmp_path` fixture.
  """
  losses = [0.5, 0.6, 0.3, 0.01, 0.55]
  with TemporaryDirectory() as d:
    tmp_path = Path(d)

    for epoch, loss in enumerate(losses):
      file = tmp_path / f"{epoch}_{loss}.pth"
      file.write_bytes(b"hello world")

    config = default_config()
    # make sure we use the temporary path
    config["save_dir"] = tmp_path.absolute().name
    config["loss_mode"] = "min"

    # test min
    path = get_best_path(tmp_path, config)
    assert path.stem.split("_")[-1] == "0.01"

    # test max
    config["loss_mode"] = "max"
    path = get_best_path(tmp_path, config)
    assert path.stem.split("_")[-1] == "0.6"
