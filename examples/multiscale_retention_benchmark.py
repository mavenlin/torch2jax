"""Benchmark MultiScaleRetention Torch vs JAX (jit) outputs with torch2jax."""

from __future__ import annotations

import time

import jax
import numpy as np
import torch
from fla.layers import MultiScaleRetention

from torch2jax import t2j


def main():
  batch_size, seq_len, hidden_size, num_heads = 2, 128, 256, 4
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise RuntimeError("flash-linear-attention kernels require a CUDA-enabled device for this benchmark.")
  dtype = torch.float16

  module = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
  module.eval()

  x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

  def torch_forward():
    with torch.no_grad():
      out, *_ = module(x)
    return out

  torch_out = torch_forward()
  torch.cuda.synchronize()
  print("torch output shape:", tuple(torch_out.shape))

  jax_module = t2j(module)
  jax_state_dict = {k: t2j(v) for k, v in module.state_dict().items()}
  jax_input = t2j(x)

  def jax_forward(jax_x):
    out = jax_module(jax_x, state_dict=jax_state_dict)
    if isinstance(out, tuple):
      out = out[0]
    return out

  jax_forward_jit = jax.jit(jax_forward)

  jax_out = jax_forward_jit(jax_input)
  jax.block_until_ready(jax_out)
  print("jax output shape:", jax_out.shape)

  torch_np = torch_out.detach().float().cpu().numpy()
  jax_np = np.array(jax.device_get(jax_out)).astype(np.float32)
  max_diff = float(np.max(np.abs(torch_np - jax_np)))
  print(f"max(|torch - jax|): {max_diff:.6f}")

  def benchmark(name, fn, sync, *args, warmup=5, iters=30):
    for _ in range(warmup):
      out = fn(*args)
      sync(out)
    start = time.perf_counter()
    for _ in range(iters):
      out = fn(*args)
      sync(out)
    elapsed = (time.perf_counter() - start) / iters
    print(f"{name:<22} {elapsed * 1e3:7.2f} ms")
    return elapsed

  def torch_sync(_):
    torch.cuda.synchronize()

  def jax_sync(out):
    jax.block_until_ready(out)

  print("\nTiming (averaged per run):")
  torch_ms = benchmark("Torch forward", lambda: torch_forward(), torch_sync)
  jax_ms = benchmark("JAX forward (jit)", lambda: jax_forward_jit(jax_input), jax_sync)
  speedup = torch_ms / jax_ms if jax_ms > 0 else float("inf")
  print(f"Speedup (Torch / JAX): {speedup:>7.3f}x")


if __name__ == "__main__":
  main()
