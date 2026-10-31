import jax
import jax.numpy as jnp
import pytest
import torch

from torch2jax import j2t, t2j

from .utils import backward_test


def test_t2j_dtype():
  assert t2j(torch.float16) == jnp.float16
  assert t2j(torch.float32) == jnp.float32
  assert t2j(torch.float64) == jnp.float64
  assert t2j(torch.int8) == jnp.int8
  assert t2j(torch.int16) == jnp.int16
  assert t2j(torch.int32) == jnp.int32
  assert t2j(torch.int64) == jnp.int64
  assert t2j(torch.uint8) == jnp.uint8
  assert t2j(torch.bool) == jnp.bool_
  assert t2j(torch.complex64) == jnp.complex64
  assert t2j(torch.complex128) == jnp.complex128
  assert t2j(torch.bfloat16) == jnp.bfloat16


def test_j2t_dtype_constructors():
  assert j2t(jnp.float16) == torch.float16
  assert j2t(jnp.float32) == torch.float32
  assert j2t(jnp.float64) == torch.float64
  assert j2t(jnp.int8) == torch.int8
  assert j2t(jnp.int16) == torch.int16
  assert j2t(jnp.int32) == torch.int32
  assert j2t(jnp.int64) == torch.int64
  assert j2t(jnp.uint8) == torch.uint8
  assert j2t(jnp.bool_) == torch.bool
  assert j2t(jnp.complex64) == torch.complex64
  assert j2t(jnp.complex128) == torch.complex128
  assert j2t(jnp.bfloat16) == torch.bfloat16


def test_j2t_dtypes():
  assert j2t(jnp.dtype("float16")) == torch.float16
  assert j2t(jnp.dtype("float32")) == torch.float32
  assert j2t(jnp.dtype("float64")) == torch.float64
  assert j2t(jnp.dtype("int8")) == torch.int8
  assert j2t(jnp.dtype("int16")) == torch.int16
  assert j2t(jnp.dtype("int32")) == torch.int32
  assert j2t(jnp.dtype("int64")) == torch.int64
  assert j2t(jnp.dtype("uint8")) == torch.uint8
  assert j2t(jnp.dtype("bool_")) == torch.bool
  assert j2t(jnp.dtype("complex64")) == torch.complex64
  assert j2t(jnp.dtype("complex128")) == torch.complex128
  assert j2t(jnp.dtype("bfloat16")) == torch.bfloat16


def test_t2j_array_conversion():
  assert t2j(torch.tensor(0.0, dtype=torch.float16)).dtype == jnp.float16
  assert t2j(torch.tensor(0.0, dtype=torch.float32)).dtype == jnp.float32
  # assert t2j(torch.tensor(0.0, dtype=torch.float64)).dtype == jnp.float64  # requires jax_enable_x64
  assert t2j(torch.tensor(0, dtype=torch.int8)).dtype == jnp.int8
  assert t2j(torch.tensor(0, dtype=torch.int16)).dtype == jnp.int16
  assert t2j(torch.tensor(0, dtype=torch.int32)).dtype == jnp.int32
  # assert t2j(torch.tensor(0, dtype=torch.int64)).dtype == jnp.int64  # requires jax_enable_x64
  assert t2j(torch.tensor(0, dtype=torch.uint8)).dtype == jnp.uint8
  assert t2j(torch.tensor(False, dtype=torch.bool)).dtype == jnp.bool_
  assert t2j(torch.tensor(0.0, dtype=torch.complex64)).dtype == jnp.complex64
  # assert t2j(torch.tensor(0.0, dtype=torch.complex128)).dtype == jnp.complex128  # requires jax_enable_x64
  assert t2j(torch.tensor(0.0, dtype=torch.bfloat16)).dtype == jnp.bfloat16


def test_j2t_array_conversion():
  assert j2t(jnp.array(0.0, dtype=jnp.float16)).dtype == torch.float16
  assert j2t(jnp.array(0.0, dtype=jnp.float32)).dtype == torch.float32
  # assert j2t(jnp.array(0.0, dtype=jnp.float64)).dtype == torch.float64  # requires jax_enable_x64
  assert j2t(jnp.array(0, dtype=jnp.int8)).dtype == torch.int8
  assert j2t(jnp.array(0, dtype=jnp.int16)).dtype == torch.int16
  assert j2t(jnp.array(0, dtype=jnp.int32)).dtype == torch.int32
  # assert j2t(jnp.array(0, dtype=jnp.int64)).dtype == torch.int64  # requires jax_enable_x64
  assert j2t(jnp.array(0, dtype=jnp.uint8)).dtype == torch.uint8
  assert j2t(jnp.array(False, dtype=jnp.bool_)).dtype == torch.bool
  assert j2t(jnp.array(0.0, dtype=jnp.complex64)).dtype == torch.complex64
  # assert j2t(jnp.array(0.0, dtype=jnp.complex128)).dtype == torch.complex128  # requires jax_enable_x64
  assert j2t(jnp.array(0.0, dtype=jnp.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize(
  "left,right",
  [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.int16, torch.float32),
    (torch.uint8, torch.int16),
  ],
)
def test_promote_types_in_converted_function(left, right):
  def add(x, y):
    return x.to(torch.promote_types(x.dtype, y.dtype)) + y

  x, y = torch.ones(3, dtype=left), torch.ones(3, dtype=right)
  expected = add(x, y)
  actual = jax.jit(t2j(add))(t2j(x), t2j(y))
  assert actual.dtype == t2j(expected.dtype)
  assert jnp.array_equal(actual, t2j(expected))


@pytest.mark.parametrize(
  "operation,left,right",
  [
    ("add_", torch.bfloat16, torch.float32),
    ("sub_", torch.bfloat16, torch.float32),
    ("mul_", torch.float16, torch.float32),
    ("div_", torch.bfloat16, torch.float32),
    ("add_", torch.uint8, torch.int32),
  ],
)
def test_inplace_arithmetic_preserves_dtype(operation, left, right):
  def call(x, y):
    x = x.clone()
    assert getattr(x, operation)(y) is x
    return x

  x = torch.tensor([[1.5, 3.5, 250], [2.0, 1.5, 3.5]], dtype=left)
  y = torch.tensor([1.003, 1 / 3, 10.03], dtype=right)
  expected = call(x, y)
  actual = jax.jit(t2j(call))(t2j(x), t2j(y))
  assert actual.dtype == t2j(left)
  assert actual.shape == expected.shape
  assert jnp.array_equal(actual, t2j(expected))


@pytest.mark.parametrize(
  "operation,left,right",
  [
    ("add_", torch.int32, torch.float32),
    ("mul_", torch.bool, torch.int32),
    ("mul_", torch.float32, torch.complex64),
    ("div_", torch.int32, torch.int32),
  ],
)
def test_inplace_arithmetic_rejects_invalid_cast(operation, left, right):
  call = lambda x, y: getattr(x, operation)(y)
  x, y = torch.ones(2, dtype=left), torch.ones(2, dtype=right)
  with pytest.raises(RuntimeError, match="cast"):
    call(x.clone(), y)
  with pytest.raises(RuntimeError, match="cast"):
    jax.jit(t2j(call))(t2j(x), t2j(y))


def test_inplace_arithmetic_rejects_expanding_destination():
  call = lambda x, y: x.add_(y)
  x, y = torch.ones(1, 3), torch.ones(2, 3)
  with pytest.raises(RuntimeError, match="broadcast"):
    call(x.clone(), y)
  with pytest.raises(RuntimeError, match="broadcast"):
    jax.jit(t2j(call))(t2j(x), t2j(y))


def test_inplace_mixed_dtype_gradient():
  call = lambda x, y: x.to(torch.bfloat16).mul_(y).add_(y).float()
  args = [jnp.array([1.5, 3.5]), jnp.array([1.003, 1 / 3])]
  backward_test(call, args, rtol=0, atol=0, strict=True)
