"""Definition of the autoencoder."""

import torch
from torch import Tensor, nn

# abs imports if can be __main__ (script)
from torch_practice.default_config import default_config
from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch.create_nn_layers import create_layers


class DynamicEncoder(nn.Module):
  """Encoder.

  Args:
      increments: list of (dim_in,dim_out) tuples.
      config: network configuration.

  """

  def __init__(
    self,
    channels: list[int],
    config: DAEConfig,
  ) -> None:
    super().__init__()
    self.config = config
    self.dense = nn.LazyLinear(config.get("latent_dimension"))
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
    dense_activation = self.config.get("dense_activ")
    conv_activation = self.config.get("c_activ")

    decoding_shapes, pool_indices = [], []

    for conv, batch in zip(self.convs, self.batch_norms, strict=False):
      # so this stores the output shape to decode to.
      decoding_shapes.append(("conv", x.size()))
      x = conv_activation(batch(conv(x)))
      x = self.dropout(x)
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
  """Create a flexible Decoder.

  Arguments:
      increments: IO_channels used to make the Encoder convolutions.

  """

  def __init__(
    self,
    channels: list[int],
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

    Arguments:
        x: input tensor
        pool_indices: these are returned by pooling, used to unpool.
        shapes: configuration for each layer, to reverse the size / computation.

    """
    dense_activation = self.config.get("dense_activ")
    conv_activation = self.config.get("c_activ")
    indices = pool_indices
    c, p = -1, -1  # convolution, pool layers tracking.
    for i in range(len(shapes) - 1, -1, -1):
      name, shape = shapes[i]
      if name == "conv":
        conv, batch = self.tconvs[c], self.batch_norms[c]
        x = batch(conv(x, output_size=shape))
        if i == 0:  # last layer
          return x
        x = self.dropout(conv_activation(x))
        c -= 1
      elif self.unpool is not None and name == "pool":
        x = self.unpool(  # type: ignore pyright bug
          x,
          indices[p],
          output_size=shape,
        )
        p -= 1
      elif name == "flatten":
        x = x.view(-1, *shape[1:])  # unflatten
      elif name == "dense":
        if self.dense is None:
          self.dense = nn.Linear(self.config.get("latent_dimension"), shape[1])
        x = self.dense(x)
        x = self.dropout(dense_activation(x))

    return x


class DynamicAE(nn.Module):
  """Auto Encoder for simple Images."""

  def __init__(
    self,
    config: DAEConfig,
  ) -> None:
    super().__init__()
    self.config = config
    # proto list of "filters" for each network.
    self.channels: list[int] = [self.config.get("in_channels")]

    # make the channels from user config.
    o_channel = config.get("init_out_channels")
    for _i in range(config.get("layers")):
      self.channels.append(o_channel)
      o_channel = int(round(o_channel * self.config.get("growth")))

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


if __name__ == "__main__":
  from torchinfo import summary

  config = default_config()  # you can tweak "config"
  model = DynamicAE(config)
  summary(model, input_size=(1, 3, 32, 32), device="cpu")
