"""Demonstrate experimental torch.autograd.Function support in torch2jax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from torch.autograd import Function

from torch2jax import t2j


class Square(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    ctx.saved_scalar = 42
    return x * x

  @staticmethod
  def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    assert ctx.saved_scalar == 42
    return grad_output * 2 * x


def torch_square(x: torch.Tensor) -> torch.Tensor:
  return Square.apply(x)


def main():
  torch_input = torch.tensor([2.0], requires_grad=True)
  out = torch_square(torch_input)
  out.backward(torch.ones_like(out))
  print("torch:", out.detach().numpy(), torch_input.grad.numpy())

  jax_square = t2j(torch_square)
  primal = jax_square(jnp.array([2.0]))
  tangent = jax.grad(lambda z: jnp.sum(jax_square(z)))(jnp.array([2.0]))
  print("jax:", primal, tangent)


if __name__ == "__main__":
  main()
