"""Ensure consistency between saving-loading."""

from datetime import datetime, timezone
from pathlib import Path

SAVEDIR_FORMAT = "%Y_%m_%d_T%H_%M_%SZ"


def assert_date_format(dirname: Path) -> None:
  """Make sure the directory is the format we chose for saving."""
  try:
    datetime.strptime(dirname.name, SAVEDIR_FORMAT).astimezone(timezone.utc)
  except ValueError as err:
    msg = f"Failed to parse format {SAVEDIR_FORMAT} in {dirname}."
    raise DirnameParsingError(msg) from err


def make_timestamp() -> str:
  """Timestamp used for directories within a run."""
  return datetime.now(timezone.utc).strftime(SAVEDIR_FORMAT)


class DirnameParsingError(Exception):
  pass
