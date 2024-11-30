"""Definition of the autoencoder."""

import torch
from torch import Tensor, nn

from torch_practice.default_config import default_config

# it seems that absolute imports are less problematic.
from torch_practice.main_types import (
  DAEConfig,
)


class DynamicEncoder(nn.Module):
  """Encoder.

  Args:
      increments: list of (dim_in,dim_out) tuples.
      config: network configuration.

  """

  def __init__(
    self,
    channels: list[tuple[int, int]],
    config: DAEConfig,
  ) -> None:
    super().__init__()
    self.config = config
    self.dense = None
    self.pool = (
      nn.MaxPool2d(
        kernel_size=config.get("p_kernel"),
        stride=config.get("p_stride"),
        return_indices=True,
      )
      if self.config.get("use_pool")
      else None
    )
    self.dropout = (
      nn.Dropout(self.config.get("dropout_rate"))
      if self.config.get("use_dropout")
      else nn.Identity()
    )
    self.convs, self.batch_norms = create_layers(
      channels_list=channels,
      is_transpose=False,
      config=config,
    )

  def forward(
    self,
    x: Tensor,
  ) -> tuple[Tensor, list[Tensor], list[tuple[str, torch.Size]]]:
    """Forward pass for Encoding.

    Returns:
        x: Tensor
        pool_indices: From the convolution
        shapes: before each pooling, and before flattening.

    """
    shapes, pool_indices = [], []
    for c, b in zip(self.convs, self.batch_norms, strict=False):
      shapes.append(("conv", x.size()))
      x = self.config.get("c_activ")(b(c(x)))
      x = self.dropout(x)
      if self.pool is not None:
        shapes.append(("pool", x.size()))
        x, index = self.pool(x)
        pool_indices.append(index)

    # Flatten and send to dense layer.
    shapes.append(("flatten", x.size()))  # unflattened size.
    x = x.view(x.size(0), -1)
    dense_i, dense_o = sum(x.shape[1:]), self.config.get("latent_dimension")
    if self.dense is None:
      self.dense = nn.Linear(dense_i, dense_o)
    x = self.config.get("dense_activ")(self.dense(x))
    shapes.append(("dense", (dense_i, dense_o)))
    x = self.dropout(x)

    return x, pool_indices, shapes


class DynamicDecoder(nn.Module):
  """Create a flexible Decoder.

  Arguments:
      increments: IO_channels used to make the Encoder convolutions.

  """

  def __init__(
    self,
    increments: list[tuple[int, int]],
    config: DAEConfig,
  ) -> None:
    super().__init__()
    self.config = config

    self.unpool = (
      nn.MaxUnpool2d(
        kernel_size=config.get("p_kernel"),
        stride=config.get("p_stride"),
      )
      if self.config.get("use_pool")
      else None
    )
    self.dropout = (
      nn.Dropout(self.config.get("dropout_rate"))
      if self.config.get("use_dropout")
      else nn.Identity()
    )
    self.dense = None

    # these aren't optional layers.
    self.tconvs, self.batch_norms = create_layers(
      channels_list=increments,
      config=config,
      is_transpose=True,
    )

  def forward(
    self,
    x: Tensor,
    pool_indices: list[Tensor],
    shapes: list[tuple[str, torch.Size]],
  ) -> Tensor:
    """Decode.

    Note: indices are not reversed. We loop backwards.

    Arguments:
        x: input tensor
        pool_indices: these are returned by pooling, used to unpool.
        shapes: configuration for each layer, to reverse the size / computation.

    """
    indices = pool_indices
    c, p = -1, -1  # convolution, pool layers tracking.
    for i in range(len(shapes) - 1, -1, -1):
      name, shape = shapes[i]
      if name == "conv":
        conv, batch = self.tconvs[c], self.batch_norms[c]
        x = batch(conv(x, output_size=shape))
        if i == 0:  # last layer
          return x
        x = self.config.get("c_activ")(x)
        x = self.dropout(x)
        c -= 1
      elif self.unpool is not None and name == "pool":
        x = self.unpool(x, indices[p], output_size=shape)
        p -= 1
      elif name == "flatten":
        x = x.view(-1, *shape[1:])  # unflatten
      elif name == "dense":
        if self.dense is None:
          self.dense = nn.Linear(shape[1], shape[0])
        x = self.config.get("dense_activ")(self.dense(x))
        x = self.dropout(x)

    return x


class DynamicAE(nn.Module):
  """Auto Encoder for simple Images."""

  def __init__(
    self,
    config: DAEConfig,
  ) -> None:
    super().__init__()
    self.increments: list[tuple[int, int]] = []
    self.config = config
    in_channels = config.get("in_channels")
    o_channels = config.get("init_out_channels")

    for _i in range(config.get("layers")):
      io_channels = (in_channels, o_channels)
      self.increments.append(io_channels)
      in_channels = o_channels
      o_channels = int(round(o_channels * self.config.get("growth")))

    self.encoder = DynamicEncoder(
      self.increments,
      self.config,
    )
    self.decoder = DynamicDecoder(self.increments, self.config)

  def forward(self, x: Tensor) -> Tensor:
    """Forward Pass for AE."""
    x_encoded, pool_indices, shapes = self.encoder(x)
    return self.decoder(x_encoded, pool_indices, shapes)


def create_layers(
  channels_list: list[tuple[int, int]],
  config: DAEConfig,
  *,
  is_transpose: bool,
) -> tuple[nn.ModuleList, nn.ModuleList]:
  """Create the list of layers.

  Arguments:
    channels_list: list of (input, output)-channel numbers.
    config: layer parameters.
    is_transpose: whether we are making the transposed convolutions.

  """
  convs = nn.ModuleList()
  batch = nn.ModuleList()
  if is_transpose:
    for in_dim, out_dim in channels_list:
      convs.append(
        nn.ConvTranspose2d(
          in_channels=out_dim,
          out_channels=in_dim,
          kernel_size=config.get("c_kernel"),
          stride=config.get("c_stride"),
        ),
      )
      batch.append(nn.BatchNorm2d(in_dim))
    return convs, batch
  for in_dim, out_dim in channels_list:
    convs.append(
      nn.Conv2d(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=config.get("c_kernel"),
        stride=config.get("c_stride"),
      ),
    )
    batch.append(nn.BatchNorm2d(out_dim))
  return convs, batch


if __name__ == "__main__":
  from torchinfo import summary

  config = default_config()  # you can tweak "config"

  model = DynamicAE(config)
  summary(model, input_size=(1, 3, 32, 32), device="cpu")
