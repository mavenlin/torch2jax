"""Utilities for routing torch.autograd.Function through torch2jax."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import jax
import jax.tree_util as jtu
import torch
import sys
import types
import warnings
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

Torchish = None
_tree_coerce: Callable = None
_ORIG_APPLY = Function.apply.__func__
_PATCH_INSTALLED = False


def _refresh_autograd_function_aliases():
  """Replace module-level aliases that still point to the original Function.apply."""
  new_apply = Function.apply
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for module in list(sys.modules.values()):
      if module is None or not hasattr(module, "__dict__"):
        continue
      attrs = vars(module)
      for name, value in list(attrs.items()):
        if isinstance(value, types.MethodType) and value.__func__ is _ORIG_APPLY:
          cls = value.__self__
          if isinstance(cls, type) and issubclass(cls, Function):
            attrs[name] = cls.apply


def init_autograd_function_support(torchish_cls, torchish_mode, tree_coerce):
  """Provide the Torchish wrapper and tree coercion helper used by torch2jax."""
  global Torchish, TorchishMode, _tree_coerce
  Torchish = torchish_cls
  TorchishMode = torchish_mode
  _tree_coerce = tree_coerce


def _tree_to_torchish(obj: Any):
  """Convert a pytree of JAX values to Torchish wrappers on demand."""
  assert Torchish is not None, "init_autograd_function_support must run first"

  def convert(x):
    if isinstance(x, Torchish):
      return x
    if isinstance(x, jax.Array):
      return Torchish(x)
    return x

  return jax.tree.map(convert, obj)


class _Ctx(FunctionCtx):
  def __init__(self):
    super().__init__()
    # self.attrs = {}

  def save_for_backward(self, *tensors):
    super().save_for_backward(*tensors)

  def mark_non_differentiable(self, *tensors):
    super().mark_non_differentiable(*tensors)

  def mark_dirty(self, *tensors):
    raise NotImplementedError(
      "torch2jax does not yet support ctx.mark_dirty inside custom autograd.Function"
    )

  def set_materialize_grads(self, value: bool):
    raise NotImplementedError(
      "torch2jax does not yet support ctx.set_materialize_grads inside custom autograd.Function"
    )

  def jax_state(self):
    return _tree_coerce(self.to_save)

  def load_state(self, state):
    self.saved_tensors = _tree_to_torchish(state)


def _make_custom_vjp(
  cls: type[Function],
  dynamic_entries: Tuple[Tuple[int, Any]],
  static_entries: Tuple[Tuple[int, Any]],
) -> Callable[..., Any]:
  dynamic_args = [v for _, v in dynamic_entries]
  dynamic_indices = [i for i, _ in dynamic_entries]
  arg_count = len(dynamic_entries) + len(static_entries)

  def _forward(*args):
    all_args = [None] * arg_count
    for idx, value in static_entries:
      all_args[idx] = value
    for idx, value in zip(dynamic_indices, args):
      all_args[idx] = value
    ctx = _Ctx()
    out = cls.forward(ctx, *_tree_to_torchish(all_args))
    return _tree_coerce(out), ctx

  # run once to obtain any static ctx
  _, _ctx = _forward(*map(jax.lax.stop_gradient, dynamic_args))

  # then construct custom vjp function
  @jax.custom_vjp
  def wrapped(*args):
    out, _ = _forward(*args)
    return out

  def fwd(*args):
    out, ctx = _forward(*args)
    return out, ctx.jax_state()

  def bwd(state, grad_output):
    _ctx.load_state(state)
    with TorchishMode():
      if not isinstance(grad_output, (tuple, list)):
        grad_output = (grad_output,)
      grads = cls.backward(_ctx, *_tree_to_torchish(grad_output))
    if not isinstance(grads, (tuple, list)):
      grads = (grads,)
    dynamic_grads = []
    for idx in dynamic_indices:
      grad = grads[idx] if idx < len(grads) else None
      dynamic_grads.append(_tree_coerce(grad) if grad is not None else None)
    return tuple(dynamic_grads)

  wrapped.defvjp(fwd, bwd)
  return wrapped


def _is_torchish_arg(values: Iterable[Any]) -> bool:
  return any(isinstance(v, Torchish) for v in values)


def _patched_apply(cls, *args, **kwargs):
  if kwargs:
    # PyTorch itself rejects kwargs; defer to the original error.
    return _ORIG_APPLY(cls, *args, **kwargs)

  if not _is_torchish_arg(args):
    return _ORIG_APPLY(cls, *args, **kwargs)

  static_entries: list[Tuple[int, Any]] = []
  dynamic_entries: list[Tuple[int, Any]] = []
  args = _tree_coerce(args)
  for idx, arg in enumerate(args):
    array_type = getattr(jax, "Array", None)
    if isinstance(array_type, type) and isinstance(arg, array_type):
      dynamic_entries.append((idx, arg))
      continue
    tracer_type = getattr(jax.core, "Tracer", None)
    if isinstance(tracer_type, type) and isinstance(arg, tracer_type):
      dynamic_entries.append((idx, arg))
    else:
      static_entries.append((idx, arg))
  # each call it needs to recapture the static values, so we won't cache it
  custom_vjp = _make_custom_vjp(cls, dynamic_entries, static_entries)
  jax_out = custom_vjp(*[v for _, v in dynamic_entries])
  return _tree_to_torchish(jax_out)


def enable_autograd_function_support():
  """Install the patched torch.autograd.Function bridge."""
  assert Torchish is not None and _tree_coerce is not None
  global _PATCH_INSTALLED
  if not _PATCH_INSTALLED:
    Function.apply = classmethod(_patched_apply)
    _refresh_autograd_function_aliases()
    _PATCH_INSTALLED = True


def disable_autograd_function_support():
  """Revert to PyTorch's original Function.apply implementation."""
  global _PATCH_INSTALLED
  if _PATCH_INSTALLED:
    Function.apply = classmethod(_ORIG_APPLY)
    _PATCH_INSTALLED = False
