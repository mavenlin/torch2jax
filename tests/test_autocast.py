from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flax import nnx

from torch2jax import Torchish, t2j

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast")


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
  "operation",
  [
    lambda x, w, b: F.linear(x, w, b),
    lambda x, w, b: x @ w,
    lambda x, w, b: x.matmul(w),
    lambda x, w, b: torch.einsum("bi,ij->bj", x, w),
    lambda x, w, b: torch.einsum("i,j->ij", b, b),
    lambda x, w, b: F.conv1d(x[None], w[:, :, None], b),
    lambda x, w, b: F.layer_norm(x.to(torch.bfloat16), (8,), b, b),
    lambda x, w, b: F.group_norm(x.to(torch.bfloat16)[None], 2, b, b),
    lambda x, w, b: F.normalize(x.to(torch.bfloat16), dim=-1),
    lambda x, w, b: x.to(torch.bfloat16).exp(),
    lambda x, w, b: x.to(torch.bfloat16) ** 2,
    lambda x, w, b: 1 / (x.to(torch.bfloat16).abs() + 1),
    lambda x, w, b: x.to(torch.bfloat16).sum(),
    lambda x, w, b: x.to(torch.bfloat16).cumsum(0),
    lambda x, w, b: F.softmax(x.to(torch.bfloat16), -1),
    lambda x, w, b: F.softmax(x.to(torch.bfloat16), -1, dtype=torch.bfloat16),
    lambda x, w, b: torch.cat((x, x.to(torch.bfloat16))),
    lambda x, w, b: F.scaled_dot_product_attention(x[None, None], w[None, None], w[None, None]),
    lambda x, w, b: F.interpolate(x.to(torch.bfloat16)[None, None], size=(4, 4), mode="nearest"),
    lambda x, w, b: F.interpolate(x.to(torch.bfloat16)[None, None], size=(4, 4), mode="bilinear"),
    lambda x, w, b: F.interpolate(x.to(torch.bfloat16)[None, None], size=(4, 4), mode="area"),
  ],
)
def test_cuda_autocast_operations(operation, dtype):
  def forward(x, w, b):
    with torch.autocast("cuda", dtype=dtype):
      return operation(x, w, b)

  args = [torch.randn(8, 8, device="cuda"), torch.randn(8, 8, device="cuda"), torch.randn(8, device="cuda")]
  expected = forward(*args)
  actual = jax.jit(t2j(forward))(*(t2j(x) for x in args))
  assert actual.dtype == t2j(expected.dtype)
  np.testing.assert_allclose(np.asarray(actual, np.float32), expected.float().cpu().numpy(), rtol=0.02, atol=0.02)


@cuda
def test_nested_scope_and_gradients():
  def forward(x, w):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      low = F.linear(x, w)
      with torch.autocast("cuda", enabled=False):
        high = F.linear(low.float(), w)
      return F.linear(high, w).float().sum()

  x, w = (torch.randn(4, 4, device="cuda", requires_grad=True) for _ in range(2))
  expected = torch.autograd.grad(forward(x, w), (x, w))
  actual = jax.jit(jax.grad(t2j(forward), argnums=(0, 1)))(t2j(x), t2j(w))
  for left, right in zip(actual, expected):
    np.testing.assert_allclose(np.asarray(left), right.cpu().numpy(), rtol=0.03, atol=0.04)


@cuda
def test_module_parameters_stay_float32():
  class Model(torch.nn.Linear):
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x):
      return super().forward(x)

  source = Model(8, 4).cuda()
  converted = t2j(source)
  result = jax.jit(converted)(t2j(torch.ones(2, 8, device="cuda")))
  assert result.dtype == jnp.bfloat16
  assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(nnx.state(converted)))
  assert source.weight.dtype == torch.float32


@cuda
def test_out_inplace_and_explicit_dtype():
  def forward(x, w):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      out = torch.zeros_like(x)
      torch.matmul(x, w, out=out)
      inplace = x.to(torch.bfloat16).exp_()
      explicit = x.to(torch.bfloat16).sum(dtype=torch.bfloat16)
      return out, inplace, explicit

  x, w = (torch.randn(4, 4, device="cuda") for _ in range(2))
  actual = jax.jit(t2j(forward))(t2j(x), t2j(w))
  for left, right in zip(actual, forward(x, w)):
    assert left.dtype == t2j(right.dtype)
    np.testing.assert_allclose(np.asarray(left, np.float32), right.float().cpu().numpy(), rtol=0.01, atol=0.01)


@cuda
def test_override_receives_autocast_arguments():
  def forward(x, w):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return F.linear(x, w)

  def linear(x, w):
    assert x.dtype == w.dtype == torch.bfloat16
    return Torchish(x.value @ w.value.T)

  converted = t2j(forward, torch_function_overrides={F.linear: linear})
  assert jax.jit(converted)(jnp.ones((2, 4)), jnp.ones((3, 4))).dtype == jnp.bfloat16


@cuda
def test_float64_and_integer_inputs_are_not_cast():
  def forward(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return x**2

  with jax.enable_x64():
    for dtype in (torch.float64, torch.int32):
      x = torch.arange(4, device="cuda", dtype=dtype)
      actual = jax.jit(t2j(forward))(t2j(x))
      assert actual.dtype == t2j(dtype)
      np.testing.assert_array_equal(actual, forward(x).cpu().numpy())


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_promotes_scatter_inputs(dtype):
  def forward(x, index, source):
    with torch.autocast("cuda", dtype=dtype):
      return x.scatter_add(0, index, source)

  args = [torch.ones(4, device="cuda"), torch.arange(4, device="cuda"), torch.ones(4, device="cuda", dtype=dtype)]
  actual = jax.jit(t2j(forward))(*(t2j(x) for x in args))
  assert actual.dtype == jnp.float32
  np.testing.assert_array_equal(actual, forward(*args).cpu().numpy())


@cuda
def test_concurrent_scopes_are_thread_local():
  barrier = Barrier(2, timeout=30)

  def trace(dtype):
    def forward(x):
      with torch.autocast("cuda", dtype=dtype):
        barrier.wait()
        return x @ x

    return jax.jit(t2j(forward))(jnp.ones((4, 4))).dtype

  with ThreadPoolExecutor(max_workers=2) as pool:
    assert list(pool.map(trace, (torch.float16, torch.bfloat16))) == [jnp.float16, jnp.bfloat16]
  assert not torch.is_autocast_enabled("cuda")

  def fail():
    with torch.autocast("cuda", dtype=torch.bfloat16):
      raise ValueError("scope failed")

  with pytest.raises(ValueError, match="scope failed"):
    t2j(fail)()
  assert not torch.is_autocast_enabled("cuda")


def test_cpu_autocast_is_not_silently_mistranslated():
  def forward(x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
      return torch.matmul(x, x)

  with pytest.raises(NotImplementedError, match="CPU autocast"):
    t2j(forward)(jnp.ones((2, 2)))
