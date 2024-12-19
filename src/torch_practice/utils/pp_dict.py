"""Format dict for pretty printing."""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from torch_practice.main_types import RunConfig
else:
  RunConfig = None


def pp_dict(
  config: RunConfig | dict,
  max_depth: int = 5,
  *,
  _indent: int = 0,
) -> str:
  """Recursively clean up config classes and names for logging.

  Args:
    config: full configuration object or dictionary.
    max_depth: how far to go in nested dictionaries
    _indent: how many spaces to indent at sublevels (internal use only.)

  """
  to_join = []
  for k, v in config.items():
    new_val = v
    new_key = k
    if isinstance(v, Callable):
      new_val = v.__name__
    elif isinstance(v, dict):
      new_key = new_key.capitalize()
      to_join.append(f"\n{new_key}\n")
      if max_depth > 0:
        to_join.append(f"{pp_dict(v,max_depth=max_depth-1, _indent=_indent+2)}")
      else:
        msg = "max_depth was reached due to deeply nested dicts."
        raise MaxDepthReachedError(msg)
      continue
    elif v is None:
      new_val = v.__class__.__name__[:4]
    elif isinstance(v, bool | str | int | float | tuple):
      new_val = v
    to_join.append(f"{new_key.rjust(_indent)} = {new_val}")
  return "\n".join(to_join)


class MaxDepthReachedError(Exception):
  pass
