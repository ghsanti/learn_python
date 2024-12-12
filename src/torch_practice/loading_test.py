"""Test io operations using tmp_path."""

from torch_practice.loading import get_best_path
from torch_practice.utils.date_format import make_timestamp


class TestGetBestPath:
  def test_best_path(self, tmp_path):
    """Create fake paths and test whether it finds it correctly."""
    max_loss = 0.6
    losses = [0.5, max_loss, 0.3, 0.01, 0.55]
    save_mode = "state_dict"
    for epoch, loss in enumerate(losses):
      # create "models" in the path.
      name = f"{save_mode}_{epoch}_{loss}.pth"
      if loss == max_loss:
        subdir = tmp_path / make_timestamp()
        subdir.mkdir(parents=False, exist_ok=False)
        file = subdir / name
      else:
        file = tmp_path / name
      file.write_bytes(b"hello world")

    # make sure we use the temporary path
    dirname = tmp_path.resolve()

    # test min and max arguments.
    path_loss = get_best_path(dirname, "min", 0, save_mode)
    assert path_loss is not None
    assert path_loss[0].stem.split("_")[-1] == "0.01"
    assert str(path_loss[1]) == "0.01"

    path_loss = get_best_path(dirname, "max", 1, save_mode)
    assert path_loss is not None
    assert path_loss[0].stem.split("_")[-1] == "0.6"
    assert str(path_loss[1]) == "0.6"

  def test_error_no_file(self, tmp_path):
    """When directory exists, but file doesn't it must error."""
    # tmp_path is unique for each fn invokation
    result = get_best_path(tmp_path.resolve(), "min", 0, "state_dict")
    assert result is None
