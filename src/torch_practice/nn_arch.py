"""Definition of the autoencoder."""

import torch
from torch import Tensor, nn

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
        kernel_size=config["c_kernel"],
        stride=config["c_stride"],
      ),
    )
    batch.append(nn.LazyBatchNorm2d())
  return convs, batch


class DynamicEncoder(nn.Module):
  def __init__(
    self,
    channels: list[int],
    config: DAEConfig,
  ) -> None:
    """Specify the Encoder.

    Args:
        channels: list of `dim_out` ints.
        config: network configuration.

    """
    super().__init__()
    self.config = config
    self.dense = nn.LazyLinear(config["latent_dimension"])
    self.pool = (
      nn.MaxPool2d(
        kernel_size=config["p_kernel"],
        stride=config["p_stride"],
        return_indices=True,
      )
      if self.config["use_pool"]
      else None
    )
    self.dropout2d = (
      nn.Dropout2d(self.config["dropout2d_rate"])
      if self.config["use_dropout2d"]
      else nn.Identity()
    )
    self.dropout = (
      nn.Dropout(self.config["dropout_rate_latent"])
      if self.config["use_dropout_latent"]
      else nn.Identity()
    )
    self.convs, self.batch_norms = create_layers(
      channels,
      config,
      is_transpose=False,
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
    dense_activation = self.config["dense_activ"]
    conv_activation = self.config["c_activ"]

    decoding_shapes, pool_indices = [], []

    for conv, batch in zip(self.convs, self.batch_norms, strict=False):
      # so this stores the output shape to decode to.
      decoding_shapes.append(("conv", x.size()))
      x = conv_activation(batch(conv(x)))
      x = self.dropout2d(x)
      if self.pool is not None:
        decoding_shapes.append(("pool", x.size()))
        x, index = self.pool(x)
        pool_indices.append(index)

    # Flatten and send to dense layer.
    decoding_shapes.append(("flatten", x.size()))  # unflattened size.
    x = x.view(x.size(0), -1)
    decoding_shapes.append(("dense", x.size()))
    x = dense_activation(self.dense(x))
    x = self.dropout(x)

    return x, pool_indices, decoding_shapes


class DynamicDecoder(nn.Module):
  def __init__(
    self,
    channels: list[int],
    config: DAEConfig,
  ) -> None:
    """Specify the Decoder.

    Args:
        channels: list of `dim_out` ints.
        config: network configuration dictionary.

    """
    super().__init__()
    self.config = config

    self.unpool = (
      nn.MaxUnpool2d(
        kernel_size=config["p_kernel"],
        stride=config["p_stride"],
      )
      if self.config["use_pool"]
      else None
    )

    self.dropout2d = (
      nn.Dropout2d(self.config["dropout2d_rate"])
      if self.config["use_dropout2d"]
      else nn.Identity()
    )
    self.dropout = (
      nn.Dropout(self.config["dropout_rate_latent"])
      if self.config["use_dropout_latent"]
      else nn.Identity()
    )

    self.dense = None

    # these aren't optional layers.
    self.tconvs, self.batch_norms = create_layers(
      channels,
      config,
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

    Args:
        x: input tensor
        pool_indices: returned by pooling, used to unpool.
        shapes: output size for each layer.

    """
    dense_activation = self.config["dense_activ"]
    conv_activation = self.config["c_activ"]
    c, p = -1, -1  # convolution, pool layers tracking.
    for i in range(len(shapes) - 1, -1, -1):
      name, shape = shapes[i]
      if name == "conv":
        conv, batch = self.tconvs[c], self.batch_norms[c]
        x = batch(conv(x, output_size=shape))
        if i == 0:  # last layer
          return x
        x = self.dropout2d(conv_activation(x))
        c -= 1
      elif self.unpool is not None and name == "pool":
        x = self.unpool(  # type: ignore pyright bug
          x,
          pool_indices[p],
          output_size=shape,
        )
        p -= 1
      elif name == "flatten":
        x = x.view(-1, *shape[1:])  # unflatten
      elif name == "dense":
        if self.dense is None:
          self.dense = nn.Linear(
            self.config["latent_dimension"],
            shape[-1],
          ).to(x.device)  # put the layer wherever the tensor is.
        x = self.dropout(dense_activation(self.dense(x)))

    return x


class DynamicAE(nn.Module):
  def __init__(
    self,
    config: DAEConfig,
  ) -> None:
    """Configure AutoEncoder.

    Args:
      config: net config dict.

    """
    super().__init__()
    self.config = config
    # proto list of "filters" for each network.
    self.channels: list[int] = []
    # append first channel, used by decoder.
    self.channels.append(self.config["input_size"][0])

    # make the channels from user config.
    o_channel = config["init_out_channels"]
    for _i in range(config["layers"]):
      self.channels.append(o_channel)
      o_channel = int(round(o_channel * self.config["growth"]))

    self.encoder = DynamicEncoder(
      self.channels[1:],  # first is the image size.
      self.config,
    )
    # discarded channel is the input "image" for the decoder.
    self.decoder = DynamicDecoder(self.channels[:-1], self.config)
    # channels are reversed but reversed makes it easier for decoding.

  def forward(self, x: Tensor) -> Tensor:
    """Forward Pass for AE."""
    x, pool_indices, decoding_shapes = self.encoder(x)
    return self.decoder(x, pool_indices, decoding_shapes)
