"""Example showing torch2jax conversion of a custom torch.autograd.Function."""

import jax
import jax.numpy as jnp
import torch
from torch.autograd import Function

from torch2jax import t2j


class Square(Function):
  @staticmethod
  def forward(ctx, input_tensor):
    ctx.save_for_backward(input_tensor)
    return input_tensor * input_tensor

  @staticmethod
  def backward(ctx, grad_output):
    (input_tensor,) = ctx.saved_tensors
    return grad_output * (2 * input_tensor)


def torch_square_with_custom_autograd(x: torch.Tensor) -> torch.Tensor:
  return Square.apply(x)


def run_demo() -> None:
  torch_input = torch.tensor([2.0], requires_grad=True)
  torch_output = torch_square_with_custom_autograd(torch_input)
  torch_output.backward(torch.ones_like(torch_output))
  print("torch output:", torch_output.detach().numpy())
  print("torch grad:", torch_input.grad.numpy())

  jax_fn = t2j(torch_square_with_custom_autograd)
  jax_output = jax_fn(jnp.array([2.0]))
  print("jax output:", jax_output)
  jax_grad = jax.grad(lambda z: jnp.sum(jax_fn(z)))(jnp.array([2.0]))
  print("jax grad:", jax_grad)


if __name__ == "__main__":
  run_demo()
