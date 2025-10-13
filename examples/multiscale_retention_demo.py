"""Convert a fla MultiScaleRetention module with torch2jax."""

from __future__ import annotations

import torch
from fla.layers import MultiScaleRetention

from torch2jax import t2j


def main():
  batch_size, seq_len, hidden_size, num_heads = 2, 128, 256, 4
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise RuntimeError("flash-linear-attention kernels require a CUDA-enabled device for this demo.")
  dtype = torch.float16

  module = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
  module.eval()

  x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

  with torch.no_grad():
    torch_out, *rest = module(x)
  print("torch output shape:", tuple(torch_out.shape))

  jax_module = t2j(module)
  jax_state_dict = {k: t2j(v) for k, v in module.state_dict().items()}
  jax_input = t2j(x)

  jax_out = jax_module(jax_input, state_dict=jax_state_dict)
  if isinstance(jax_out, tuple):
    jax_out = jax_out[0]
  print("jax output shape:", jax_out.shape)
  print(jax_out)


if __name__ == "__main__":
  main()
