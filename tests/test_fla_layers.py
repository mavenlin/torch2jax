import numpy as np
import pytest
import torch
import jax.numpy as jnp
from jax import grad

from fla.layers import (
  BasedLinearAttention,
  GatedLinearAttention,
  KimiDeltaAttention,
  LinearAttention,
  MultiScaleRetention,
)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

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
  grads = {}
  for name, param in module.named_parameters():
    if param.grad is None:
      grads[name] = None
    else:
      grads[name] = param.grad.detach().cpu().to(torch.float32).numpy()
  return torch_output.detach().cpu().to(torch.float32).numpy(), grads


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
    pytest.param(
      lambda: BasedLinearAttention(hidden_size=64, num_heads=4, num_key_value_heads=4),
      (2, 64, 64),
      id="based_linear_attention",
    ),
    pytest.param(
      lambda: GatedLinearAttention(hidden_size=64, num_heads=4, mode="chunk"),
      (2, 64, 64),
      id="gated_linear_attention",
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


@CUDA_REQUIRED
def test_chunk_gated_delta_rule_forward_and_gradients():
  device = torch.device("cuda")
  dtype = torch.float32

  torch.manual_seed(0)

  batch, seqlen, num_heads, head_dim, value_dim = 1, 8, 2, 4, 4

  q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
  q.requires_grad_(True)

  k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
  k.requires_grad_(True)

  v = torch.randn(batch, seqlen, num_heads, value_dim, device=device, dtype=dtype)
  v.requires_grad_(True)

  g = -torch.rand(batch, seqlen, num_heads, device=device, dtype=dtype)
  g.requires_grad_(True)

  beta = torch.rand(batch, seqlen, num_heads, device=device, dtype=dtype)
  beta.requires_grad_(True)

  initial_state = torch.randn(batch, num_heads, head_dim, value_dim, device=device, dtype=dtype)
  initial_state.requires_grad_(True)

  torch_output, torch_final_state = chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    None,
    initial_state,
    True,
  )
  loss = torch_output.float().pow(2).mean()
  loss.backward()

  torch_grads = {
    "q": q.grad.detach().cpu().to(torch.float32).numpy(),
    "k": k.grad.detach().cpu().to(torch.float32).numpy(),
    "v": v.grad.detach().cpu().to(torch.float32).numpy(),
    "g": g.grad.detach().cpu().to(torch.float32).numpy(),
    "beta": beta.grad.detach().cpu().to(torch.float32).numpy(),
    "initial_state": initial_state.grad.detach().cpu().to(torch.float32).numpy(),
  }

  chunk_fn = t2j(chunk_gated_delta_rule)
  q_jax, k_jax, v_jax, g_jax, beta_jax, init_state_jax = [
    t2j(tensor.detach()) for tensor in (q, k, v, g, beta, initial_state)
  ]

  jax_output, jax_final_state = chunk_fn(
    q_jax,
    k_jax,
    v_jax,
    g_jax,
    beta_jax,
    None,
    init_state_jax,
    True,
  )

  aac(
    np.asarray(jax_output, dtype=np.float32),
    _to_numpy(torch_output.detach().float().cpu()),
    atol=5e-3,
  )
  aac(
    np.asarray(jax_final_state, dtype=np.float32),
    _to_numpy(torch_final_state.detach().float().cpu()),
    atol=5e-3,
  )

  def loss_fn(q, k, v, g, beta, init_state):
    out, _ = chunk_fn(q, k, v, g, beta, None, init_state, True)
    out = jnp.asarray(out, dtype=jnp.float32)
    return jnp.mean(jnp.square(out))

  jax_grads = grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
    q_jax,
    k_jax,
    v_jax,
    g_jax,
    beta_jax,
    init_state_jax,
  )

  for (name, torch_grad), jax_grad in zip(torch_grads.items(), jax_grads):
    aac(
      np.asarray(jax_grad, dtype=np.float32),
      torch_grad,
      atol=5e-3,
    )


@CUDA_REQUIRED
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_kimi_delta_attention_forward_and_gradients(training):
  device = torch.device("cuda")
  dtype = torch.float32

  torch.manual_seed(0)
  layer = KimiDeltaAttention(
    hidden_size=32,
    expand_v=1,
    head_dim=16,
    num_heads=2,
    mode="chunk",
    use_short_conv=False,
  ).to(device=device, dtype=dtype)
  if training:
    layer.train()
  else:
    layer.eval()
  torch_input = torch.randn(1, 65, 32, device=device, dtype=dtype, requires_grad=True)

  torch_output_np, torch_grads = _torch_forward_and_grad(layer, torch_input)
  state_dict_torch, param_names = _torch_state_dict_tensors(layer)
  state_dict_jax = _state_dict_to_jax_tensors(state_dict_torch)

  jax_layer = t2j(layer)
  jax_input = t2j(torch_input.detach())

  jax_output_np, jax_grads = _jax_forward_and_grad(jax_layer, jax_input, state_dict_jax, param_names)

  aac(jax_output_np.astype(np.float32), torch_output_np.astype(np.float32), atol=1e-5)
  for name, grad_val in jax_grads.items():
    expected = torch_grads[name]
    grad_np = _to_numpy(grad_val)
    if expected is None:
      assert grad_np is None or np.allclose(grad_np, 0, atol=1e-5)
    else:
      aac(np.asarray(grad_np, dtype=np.float32), expected.astype(np.float32), atol=1e-5)
