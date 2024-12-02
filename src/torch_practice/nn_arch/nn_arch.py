"""Definition of the autoencoder."""

import torch
from torch import Tensor, nn

# abs imports if can be __main__ (script)
from torch_practice.default_config import default_config
from torch_practice.main_types import DAEConfig
from torch_practice.nn_arch.create_nn_layers import create_layers


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
    self.dropout2d = (
      nn.Dropout2d(self.config.get("dropout_rate"))
      if self.config.get("use_dropout")
      else nn.Identity()
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
        kernel_size=config.get("p_kernel"),
        stride=config.get("p_stride"),
      )
      if self.config.get("use_pool")
      else None
    )
    self.dropout2d = (
      nn.Dropout2d(self.config.get("dropout_rate"))
      if self.config.get("use_dropout")
      else nn.Identity()
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

    Args:
        x: input tensor
        pool_indices: returned by pooling, used to unpool.
        shapes: output size for each layer.

    """
    dense_activation = self.config.get("dense_activ")
    conv_activation = self.config.get("c_activ")
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
            self.config.get("latent_dimension"),
            shape[-1],
          ).to(x.device)
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
  import logging

  from torchinfo import summary

  lgr = logging.getLogger()
  logging.basicConfig(level="DEBUG")  # default is warn

  config = default_config()  # you can tweak "config"
  model = DynamicAE(config)
  summary(model, input_size=(1, 3, 32, 32), device="cpu")

  # simple profiling info
  from torch.profiler import ProfilerActivity, profile, record_function

  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.xpu.is_available():
    device = "xpu"
  elif torch.mps.is_available():
    device = "mps"

  lgr.debug("Device %s", device)
  if device == "mps":
    import sys

    # torch.mps.profiler.profile(mode='interval', wait_until_completed=False)
    # disabled for now as I can't test.
    lgr.critical("MPS profiling is disabled.")
    sys.exit(0)
  else:
    img = torch.randn(config.get("batch_size"), 3, 32, 32).to(device)
    model = model.to(device)
    with (
      profile(
        activities=[
          ProfilerActivity.CPU,
          # ProfilerActivity.XPU,
          # ProfilerActivity.CUDA,
          # ProfilerActivity.MPS # may be available in the future.
        ],
        record_shapes=True,
        profile_memory=True,
      ) as prof,
      record_function("model_inference"),
    ):
      # with
      model(img)
    lgr.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
