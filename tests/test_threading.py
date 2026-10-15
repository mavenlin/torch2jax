from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from torch2jax import Torchish, t2j


def test_concurrent_traces_isolate_overrides_and_rng():
  barrier = Barrier(2, timeout=30)

  def trace(seed):
    def forward():
      barrier.wait()
      return torch.arange(4) + torch.zeros(4) + torch.rand(4)

    def arange(n):
      return Torchish(jnp.arange(n, dtype=jnp.float32) + 10 * seed)

    return jax.jit(t2j(forward, torch_function_overrides={torch.arange: arange}))(rng=jax.random.key(seed))

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(trace, (0, 1)))

  for seed, result in enumerate(results):
    expected = jnp.arange(4) + 10 * seed + jax.random.uniform(jax.random.split(jax.random.key(seed))[1], (4,))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_conversion_does_not_modify_other_threads_torch():
  entered, release = Event(), Event()
  originals = torch.arange, torch.zeros, torch.Tensor.__new__

  def forward():
    entered.set()
    assert release.wait(timeout=30)
    raise RuntimeError("conversion failed")

  with ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(t2j(forward))
    try:
      assert entered.wait(timeout=30)
      assert (torch.arange, torch.zeros, torch.Tensor.__new__) == originals
      assert isinstance(torch.arange(4), torch.Tensor)
      assert isinstance(torch.Tensor([1, 2]), torch.Tensor)
    finally:
      release.set()
    with pytest.raises(RuntimeError, match="conversion failed"):
      future.result(timeout=30)

  assert (torch.arange, torch.zeros, torch.Tensor.__new__) == originals
  assert isinstance(torch.zeros(4), torch.Tensor)


def test_nested_conversion_restores_constructor_overrides_on_error():
  def fail():
    np.testing.assert_array_equal(torch.arange(2).value, jnp.full((2,), 3))
    raise RuntimeError("inner conversion failed")

  inner = t2j(fail, torch_function_overrides={torch.arange: lambda n: Torchish(jnp.full((n,), 3))})

  def outer():
    with pytest.raises(RuntimeError, match="inner conversion failed"):
      inner()
    return torch.arange(2)

  converted = t2j(outer, torch_function_overrides={torch.arange: lambda n: Torchish(jnp.full((n,), 2))})
  np.testing.assert_array_equal(converted(), jnp.full((2,), 2))
  assert isinstance(torch.arange(2), torch.Tensor)
