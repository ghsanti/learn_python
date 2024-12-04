"""Create layers from description."""

from torch import nn

# abs imports if can be __main__ (script)
from torch_practice.main_types import DAEConfig


def create_layers(
  channels: list[int],
  config: DAEConfig,
  *,
  is_transpose: bool,
) -> tuple[nn.ModuleList, nn.ModuleList]:
  """Create list of layers from channels.

  Arguments:
    channels: list of (input, output)-channel numbers.
    config: layer parameters.
    is_transpose: whether we are making the transposed convolutions.

  """
  convs = nn.ModuleList()
  batch = nn.ModuleList()
  layer = nn.LazyConvTranspose2d if is_transpose else nn.LazyConv2d
  for in_dim in channels:
    convs.append(
      layer(
        out_channels=in_dim,
        kernel_size=config.get("c_kernel"),
        stride=config.get("c_stride"),
      ),
    )
    batch.append(nn.LazyBatchNorm2d())
  return convs, batch
