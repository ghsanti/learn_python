"""Handle model loading."""

import logging
import re
from pathlib import Path
from typing import NamedTuple

import torch

from torch_practice.main_types import LossModeType
from torch_practice.nn_arch import DynamicAE
from torch_practice.saving import SaveModeType
from torch_practice.utils.date_format import (
  DirnameParsingError,
  assert_date_format,
)
from torch_practice.utils.track_loss import loss_improved

logger = logging.getLogger(__package__)

LOSS_PATTERN = r".*_(\d+\.\d+)\.pth?$"


class Loader:
  def __init__(
    self,
    model_mode: SaveModeType,
  ) -> None:
    """Load model or checkpoint.

    Utility class that goes hand in hand with Save.

    Args:
      model_mode: specify the type of saving carried out earlier.


    """
    self.model_mode = model_mode
    self.assert_date_format = assert_date_format

  def from_loss_mode(
    self,
    from_dir: Path,
    mode: LossModeType,
    net: DynamicAE | None,
    *,
    descend_one: bool,
  ) -> NamedTuple:
    """Load the state dict(s) of the saved model.

    Args:
      from_dir: base directory to search from.
      mode: the loss modes available.
      net: instance of the model. Not needed for full model loading.
      descend_one: if True it descends to *first children only.*

    Returns:
      NamedTuple. _On inference, the model is mutated._

      Note that, as long as you pass a different instance, it loads a new model.
      So you can use the same loader for different models.

    """
    depth = 1 if descend_one else 0
    logger.debug("finding best model...")
    # recurses once if depth = 1
    result = get_best_path(from_dir, mode, depth)

    if result is None:
      log = "Use `logger.basicConfig(level='DEBUG'[, force=True])` if needed."
      logger.warning(log)
      msg = f"Could not find a filename under {from_dir}"
      raise FileNotFoundError(msg)

    fullname, loss = result
    msg = f"Found {fullname}, and loss {loss}."
    logger.info(msg)

    return self.from_filename(fullname, net)

  def from_filename(
    self,
    path_to_model: Path,
    net: DynamicAE | None,
  ) -> NamedTuple:
    """Checkpoint loading.

    Args:
      path_to_model: path to the saved `.pth` file.
      net: instance of the model. Not needed for full model loading.

    Returns:
      Named tuple of stateful items (Model, Loss, Epoch, Optimizer.)

    """
    checkpoint = torch.load(path_to_model, weights_only=True)
    if self.model_mode == "inference":
      if net is not None:
        msg = f"Loading the state dict into {net.__class__.__name__}"
        logger.info(msg)
        return net.load_state_dict(checkpoint)
      msg = "for inference mode, a model instance must be passed."
      raise ValueError(msg)
    logger.info("Returning checkpoint.")
    return checkpoint


class LossNotFoundError(Exception):
  pass


def extract_compare(
  file: Path,
  pattern: str,
  best_loss: float | None,
  mode: LossModeType,
) -> tuple[Path, float] | None:
  """Compare extracted vs best loss.

  The extraction is from the filenames `epoch_loss.pth`.
  """
  # could use stem and different pattern. match as below seems safer.
  match = re.match(pattern, file.name)
  logger.debug("Found model file.")
  if match:
    msg = f"Found match from regex ({match.group(1)})."
    logger.debug(msg)
    loss = float(match.group(1))
    improved = loss_improved(best_loss, loss, mode)
    if improved:
      return file, loss
  return None


def get_best_path(
  save_dir: Path,
  mode: LossModeType,
  depth: int,
) -> tuple[Path, float] | None:
  """Find best model if filename is not specified by user.

  The function uses a regex/pattern to find a float in the filename.
  If it doesn't, it throws a "LossNotFoundError."
  """
  # match files like `abc_0.124.pth` (or `.pt`)
  loss_pattern = LOSS_PATTERN
  best_name, best_loss = None, None

  for file in save_dir.iterdir():
    if file.is_file():
      result = extract_compare(file, loss_pattern, best_loss, mode)
      if result is not None:
        best_name, best_loss = result
    elif file.is_dir():
      try:  # only parse timestamped directories.
        assert_date_format(file)
        if depth > 0:
          depth -= 1
          result = get_best_path(file, mode, depth)
          if result is not None and loss_improved(best_loss, result[1], mode):
            best_name, best_loss = result
      except DirnameParsingError as err:
        msg = f"Not a timestamped directory {file}. Error {err}"
        logger.debug(msg)
        continue

  if best_name is not None and best_loss is not None:
    return best_name, best_loss  # full path
  return None
