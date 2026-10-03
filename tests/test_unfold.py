import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from einops import rearrange

from torch2jax import t2j


@pytest.mark.parametrize(
  "shape,dim,size,step",
  [
    ((2, 7, 3), 1, 2, 2),
    ((2, 7, 3), -2, 3, 1),
    ((2, 7, 3), 1, 2, 3),
    ((2, 7, 3), 1, 0, 2),
    ((0, 7), 1, 2, 2),
    ((2, 0), 1, 0, 1),
    ((), 0, 1, 1),
    ((), -1, 0, 1),
  ],
)
def test_unfold_forward_and_gradient(shape, dim, size, step):
  values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
  source = torch.tensor(values, requires_grad=True)
  expected = source.unfold(dim, size, step)
  converted = t2j(lambda x: x.unfold(dim, size, step))
  np.testing.assert_array_equal(jax.jit(converted)(jnp.asarray(values)), expected.detach().numpy())
  expected.square().sum().backward()
  actual_grad = jax.jit(jax.grad(lambda x: jnp.square(converted(x)).sum()))(jnp.asarray(values))
  np.testing.assert_array_equal(actual_grad, source.grad.numpy())


@pytest.mark.parametrize(
  "dim,size,step,error",
  [
    (2, 1, 1, IndexError),
    (-3, 1, 1, IndexError),
    (1, -1, 1, RuntimeError),
    (1, 4, 1, RuntimeError),
    (1, 1, 0, RuntimeError),
    (1, 1, -1, RuntimeError),
    (1.0, 1, 1, TypeError),
    (1, 1.0, 1, TypeError),
    (1, 1, 1.0, TypeError),
  ],
)
def test_unfold_invalid_arguments(dim, size, step, error):
  with pytest.raises(error):
    torch.ones((2, 3)).unfold(dim, size, step)
  with pytest.raises(error):
    t2j(lambda x: x.unfold(dim, size, step))(jnp.ones((2, 3)))


def test_nonoverlapping_video_patches_use_layout_operations():
  converted = t2j(lambda x: x.unfold(2, 2, 2).unfold(3, 3, 3).unfold(4, 3, 3))
  source = torch.arange(2 * 3 * 5 * 7 * 7).reshape((2, 3, 5, 7, 7))
  values = jnp.asarray(source.numpy())
  np.testing.assert_array_equal(
    jax.jit(converted)(values), source.unfold(2, 2, 2).unfold(3, 3, 3).unfold(4, 3, 3).numpy()
  )
  primitives = {equation.primitive.name for equation in jax.make_jaxpr(converted)(values).jaxpr.eqns}
  assert primitives <= {"slice", "reshape", "transpose"}


def test_unfold_rearrange_noncontiguous_input():
  def patches(x):
    x = x.transpose(3, 4).unfold(2, 2, 2).unfold(3, 3, 3).unfold(4, 3, 3)
    return rearrange(x, "b c t h w dt dh dw -> b (t h w) (c dt dh dw)")

  source = torch.arange(2 * 3 * 5 * 7 * 7, dtype=torch.float32).reshape(2, 3, 5, 7, 7)
  source.requires_grad_(True)
  expected = patches(source)
  values = jnp.asarray(source.detach().numpy())
  converted = t2j(patches)
  np.testing.assert_array_equal(jax.jit(converted)(values), expected.detach().numpy())
  expected.square().sum().backward()
  np.testing.assert_array_equal(jax.jit(jax.grad(lambda x: jnp.square(converted(x)).sum()))(values), source.grad)
