import numpy as np
import pytest
import torch
import jax.numpy as jnp
from jax import grad

from fla.layers import LinearAttention, MultiScaleRetention

from torch2jax import t2j

from .utils import aac


CUDA_REQUIRED = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="FLA kernels require a CUDA-enabled device.",
)


def _to_numpy(array):
  if array is None:
    return None
  return np.asarray(np.array(array))


def _torch_state_dict_tensors(module):
  state_dict = {name: tensor.detach() for name, tensor in module.state_dict().items()}
  param_names = tuple(name for name, _ in module.named_parameters())
  return state_dict, param_names


def _state_dict_to_jax_tensors(state_dict):
  return {name: t2j(tensor) for name, tensor in state_dict.items()}


def _torch_forward_and_grad(module, inputs):
  module.zero_grad(set_to_none=True)
  torch_output = module(inputs)
  if isinstance(torch_output, tuple):
    torch_output = torch_output[0]
  loss = torch_output.pow(2).mean()
  loss.backward()
  grads = {
    name: (param.grad.detach().cpu().numpy() if param.grad is not None else None)
    for name, param in module.named_parameters()
  }
  return torch_output.detach().cpu().numpy(), grads


def _jax_forward_and_grad(jax_module, jax_input, state_dict, param_names):
  base_state_dict = dict(state_dict)
  params = {name: base_state_dict[name] for name in param_names}

  def run_module(p):
    sd = dict(base_state_dict)
    sd.update(p)
    output = jax_module(jax_input, state_dict=sd)
    if isinstance(output, tuple):
      output = output[0]
    return output

  output = run_module(params)

  def loss_fn(p):
    out = run_module(p)
    return jnp.mean(jnp.square(out))

  grads = grad(loss_fn)(params)
  return np.asarray(output), grads


@CUDA_REQUIRED
@pytest.mark.parametrize(
  ("layer_ctor", "input_shape"),
  [
    pytest.param(
      lambda: LinearAttention(hidden_size=64, num_heads=4, mode="chunk"),
      (2, 64, 64),
      id="linear_attention",
    ),
    pytest.param(
      lambda: MultiScaleRetention(hidden_size=64, num_heads=4, mode="chunk"),
      (2, 64, 64),
      id="multiscale_retention",
    ),
  ],
)
def test_fla_layers_forward_and_gradients(layer_ctor, input_shape):
  device = torch.device("cuda")
  dtype = torch.float32

  torch.manual_seed(0)
  module = layer_ctor().to(device=device, dtype=dtype)
  module.train()

  torch_input = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)

  torch_output_np, torch_grads = _torch_forward_and_grad(module, torch_input)

  state_dict_torch, param_names = _torch_state_dict_tensors(module)
  state_dict_jax = _state_dict_to_jax_tensors(state_dict_torch)

  jax_module = t2j(module)
  jax_input = t2j(torch_input.detach())

  jax_output_np, jax_grads = _jax_forward_and_grad(jax_module, jax_input, state_dict_jax, param_names)

  # aac(jax_output_np, torch_output_np, atol=1e-5)

  # for name, grad_val in jax_grads.items():
  #   expected = torch_grads[name]
  #   grad_np = _to_numpy(grad_val)
  #   if expected is None:
  #     assert grad_np is None or np.allclose(grad_np, 0, atol=1e-5)
  #   else:
  #     aac(grad_np, expected, atol=1e-5)
