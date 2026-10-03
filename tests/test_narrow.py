import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from torch2jax import t2j


@pytest.mark.parametrize("dim,start,length", [(1, 1, 2), (-1, -2, 2), (1, 4, 0)])
def test_narrow_forward_and_gradient(dim, start, length):
  source = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
  expected = source.narrow(dim, start, length)
  converted = t2j(lambda x: x.narrow(dim, start, length))
  values = jnp.asarray(source.detach().numpy())
  np.testing.assert_array_equal(jax.jit(converted)(values), expected.detach().numpy())
  expected.square().sum().backward()
  np.testing.assert_array_equal(jax.grad(lambda x: jnp.square(converted(x)).sum())(values), source.grad)
