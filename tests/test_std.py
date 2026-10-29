import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from torch2jax import t2j

_OPTIONS = [
  {},
  dict(dim=0),
  dict(dim=-1, keepdim=True),
  dict(dim=(-2, -1), correction=0, keepdim=True),
  dict(dim=[0, 2], correction=2.5),
  dict(dim=(0, 1, 2), correction=0),
  dict(unbiased=False),
  dict(dim=1, unbiased=False, keepdim=True),
]


def _numpy(value):
  value = value.detach()
  return (value.float() if value.dtype == torch.bfloat16 else value).numpy()


@pytest.mark.parametrize("options", _OPTIONS)
@pytest.mark.parametrize("member", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.float16, jnp.bfloat16])
def test_std_forward_and_gradient(options, member, dtype):
  values = np.random.default_rng(0).normal(size=(2, 3, 4))
  if jnp.issubdtype(dtype, jnp.complexfloating):
    values = values + 1j * np.random.default_rng(1).normal(size=values.shape)
  torch_dtype = {
    jnp.float32: torch.float32,
    jnp.float64: torch.float64,
    jnp.complex64: torch.complex64,
    jnp.float16: torch.float16,
    jnp.bfloat16: torch.bfloat16,
  }[dtype]
  source = torch.tensor(values, dtype=torch_dtype, requires_grad=True)

  def call(x):
    return x.std(**options) if member else torch.std(x, **options)

  expected = call(source)
  expected.square().sum().backward()
  with jax.enable_x64():
    inputs = jnp.asarray(values, dtype=dtype)
    converted = t2j(call)
    actual = jax.jit(converted)(inputs)
    gradient = jax.jit(jax.grad(lambda x: jnp.square(converted(x)).sum()))(inputs)
    if jnp.issubdtype(dtype, jnp.complexfloating):
      gradient = jnp.conj(gradient)
    tolerance = 0.03 if dtype == jnp.bfloat16 else 0.003 if dtype == jnp.float16 else 2e-5
    np.testing.assert_allclose(actual.astype(jnp.float64), _numpy(expected), rtol=tolerance, atol=tolerance)
    gradient_dtype = jnp.complex64 if dtype == jnp.complex64 else jnp.float64
    np.testing.assert_allclose(gradient.astype(gradient_dtype), _numpy(source.grad), rtol=tolerance, atol=tolerance)
    assert actual.dtype == inputs.real.dtype


@pytest.mark.parametrize("correction", [0, 1])
def test_std_constant_gradient(correction):
  source = torch.ones((2, 4), requires_grad=True)
  call = lambda x: x.std(dim=-1, correction=correction).sum()
  call(source).backward()
  inputs = jnp.ones((2, 4))
  np.testing.assert_array_equal(jax.jit(t2j(call))(inputs), 0)
  np.testing.assert_array_equal(jax.jit(jax.grad(t2j(call)))(inputs), source.grad.numpy())


@pytest.mark.parametrize("values,correction", [([1.0], 1), ([1.0, 2.0], 3), ([], 1)])
def test_std_degenerate_reduction(values, correction):
  source = torch.tensor(values, dtype=torch.float32)
  call = lambda x: torch.std(x, correction=correction)
  with pytest.warns(UserWarning, match="degrees of freedom"):
    expected = call(source)
  np.testing.assert_allclose(
    jax.jit(t2j(call))(jnp.asarray(values, dtype=jnp.float32)), expected.numpy(), equal_nan=True
  )


def test_std_out_and_legacy_positional_unbiased():
  def call(x):
    out = torch.zeros(())
    returned = torch.std(x, dim=None, correction=0, out=out)
    assert returned is out
    return out, x.std(False)

  source = torch.arange(6, dtype=torch.float32)
  expected = call(source)
  actual = jax.jit(t2j(call))(jnp.arange(6, dtype=jnp.float32))
  np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_std_rejects_unbiased_with_correction():
  with pytest.raises(TypeError):
    t2j(lambda x: x.std(unbiased=True, correction=0))(jnp.ones(4))


def test_std_rejects_integer_input():
  with pytest.raises(RuntimeError, match="floating point and complex"):
    t2j(torch.std)(jnp.ones(4, dtype=jnp.int32))


@pytest.mark.parametrize("dim", [None, 0, -1])
def test_std_scalar(dim):
  call = lambda x: x.std(dim=dim, correction=0)
  np.testing.assert_array_equal(jax.jit(t2j(call))(jnp.float32(2)), call(torch.tensor(2.0)).numpy())


@pytest.mark.parametrize("dim", [(), []])
def test_std_empty_dims(dim):
  inputs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
  call = lambda x: x.std(dim=dim, correction=0)
  source = torch.tensor(np.asarray(inputs), requires_grad=True)
  np.testing.assert_allclose(jax.jit(t2j(call))(inputs), _numpy(call(source)), rtol=1e-6)
  # Torch's empty-dim backward omits the reduction count; explicit axes give
  # the derivative of the same all-elements standard deviation.
  source.std(dim=(0, 1), correction=0).backward()
  np.testing.assert_allclose(jax.jit(jax.grad(t2j(call)))(inputs), _numpy(source.grad), rtol=1e-6)
