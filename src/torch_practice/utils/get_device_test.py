import torch

from torch_practice.utils.get_device import get_device


def mock_available() -> bool:
  return True


def mock_unavailable() -> bool:
  return False


def test_cuda_device(monkeypatch) -> None:
  # we just set the attribute on the torch object.
  m = monkeypatch
  m.setattr(torch.cuda, "is_available", mock_available)
  m.setattr(torch.mps, "is_available", mock_unavailable)
  m.setattr(torch.cpu, "is_available", mock_unavailable)
  result = get_device()
  assert result == "cuda"

  m.setattr(torch.cuda, "is_available", mock_unavailable)
  m.setattr(torch.mps, "is_available", mock_available)
  result = get_device()
  assert result == "mps"

  m.setattr(torch.mps, "is_available", mock_unavailable)
  m.setattr(torch.cpu, "is_available", mock_available)
  result = get_device()
  assert result == "cpu"
