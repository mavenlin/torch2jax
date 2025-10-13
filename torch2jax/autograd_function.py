"""Utilities for routing torch.autograd.Function through torch2jax."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import jax
import jax.tree_util as jtu
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

TORCHISH = None
_TREE_COERCE: Callable = None
_ORIG_APPLY = Function.apply.__func__
_PATCH_INSTALLED = False


def init_autograd_function_support(torchish_cls, tree_coerce: Callable):
  """Provide the Torchish wrapper and tree coercion helper used by torch2jax."""
  global _TORCHISH, _TREE_COERCE
  _TORCHISH = torchish_cls
  _TREE_COERCE = tree_coerce


def _tree_to_torchish(obj: Any):
  """Convert a pytree of JAX values to Torchish wrappers on demand."""
  assert _TORCHISH is not None, "init_autograd_function_support must run first"

  def convert(x):
    if isinstance(x, _TORCHISH):
      return x
    if isinstance(x, jax.Array):
      return _TORCHISH(x)
    return x

  return jax.tree.map(convert, obj)


class _CapturedCtx(FunctionCtx):
  _INTERNAL_FIELDS = {
    "needs_input_grad",
    "to_save",
    "non_differentiable",
    "dirty_tensors",
  }

  def __init__(self, num_inputs: int):
    super().__init__()
    self.needs_input_grad = (True,) * num_inputs
    self.to_save: Tuple[Any, ...] = ()
    self.non_differentiable: Tuple[Any, ...] = ()
    self.dirty_tensors: Tuple[Any, ...] = ()

  def save_for_backward(self, *tensors):
    super().save_for_backward(*tensors)
    self.to_save = tensors

  def mark_non_differentiable(self, *tensors):
    super().mark_non_differentiable(*tensors)
    self.non_differentiable = tensors

  def mark_dirty(self, *tensors):
    raise NotImplementedError(
      "torch2jax does not yet support ctx.mark_dirty inside custom autograd.Function"
    )

  def set_materialize_grads(self, value: bool):
    raise NotImplementedError(
      "torch2jax does not yet support ctx.set_materialize_grads inside custom autograd.Function"
    )

  def export_state(self):
    return {
      "to_save": self.to_save,
      "non_differentiable": self.non_differentiable,
      "dirty_tensors": self.dirty_tensors,
      "needs_input_grad": self.needs_input_grad,
      "attrs": {
        name: getattr(self, name)
        for name in self.__dict__
        if name not in self._INTERNAL_FIELDS and not name.startswith("_")
      },
    }

  @classmethod
  def from_state(cls, num_inputs: int, state):
    ctx = cls(num_inputs)
    ctx.needs_input_grad = state["needs_input_grad"]
    ctx.to_save = tuple(_tree_to_torchish(t) for t in state["to_save"])
    ctx.non_differentiable = tuple(
      _tree_to_torchish(t) for t in state["non_differentiable"]
    )
    ctx.dirty_tensors = tuple(
      _tree_to_torchish(t) for t in state["dirty_tensors"]
    )
    ctx.saved_tensors = ctx.to_save
    for name, value in state["attrs"].items():
      setattr(ctx, name, _tree_to_torchish(value))
    return ctx


def _make_custom_vjp(
  cls: type[Function],
  arg_count: int,
  dynamic_indices: Tuple[int],
  static_entries: Tuple[Tuple[int, Any]],
) -> Callable[..., Any]:
  assert _TREE_COERCE is not None
  tree_coerce = _TREE_COERCE

  def _forward(*dynamic_args):
    args = [None] * arg_count
    for idx, value in static_entries:
      args[idx] = value
    for idx, value in zip(dynamic_indices, dynamic_args):
      args[idx] = value
    torchish_args = _tree_to_torchish(args)
    ctx = _CapturedCtx(len(torchish_args))
    out = cls.forward(ctx, *torchish_args)
    return tree_coerce(out), tree_coerce(ctx.export_state())

  @jax.custom_vjp
  def wrapped(*dynamic_args):
    out, _ = _forward(*dynamic_args)
    return out

  def fwd(*dynamic_args):
    out, state = _forward(*dynamic_args)
    return out, state

  def bwd(state, grad_output):
    ctx = _CapturedCtx.from_state(arg_count, _tree_to_torchish(state))
    torchish_grad_output = _tree_to_torchish(grad_output)
    grads = cls.backward(ctx, torchish_grad_output)
    if not isinstance(grads, (tuple, list)):
      grads = (grads,)
    grads = list(grads)
    dynamic_grads = []
    for idx in dynamic_indices:
      grad = grads[idx] if idx < len(grads) else None
      dynamic_grads.append(tree_coerce(grad) if grad is not None else None)
    return tuple(dynamic_grads)

  wrapped.defvjp(fwd, bwd)
  return wrapped


def _is_torchish_arg(values: Iterable[Any]) -> bool:
  return any(isinstance(v, _TORCHISH) for v in values)


def _patched_apply(cls, *args, **kwargs):
  if kwargs:
    # PyTorch itself rejects kwargs; defer to the original error.
    return _ORIG_APPLY(cls, *args, **kwargs)

  if not _is_torchish_arg(args):
    return _ORIG_APPLY(cls, *args, **kwargs)

  assert _TREE_COERCE is not None
  tree_coerce = _TREE_COERCE
  arg_count = len(args)
  static_entries: list[Tuple[int, Any]] = []
  dynamic_indices: list[int] = []
  dynamic_values: list[jax.Array] = []
  args = tree_coerce(args)
  for idx, arg in enumerate(args):
    if isinstance(arg, jax.Array):
      dynamic_indices.append(idx)
      dynamic_values.append(arg)
    else:
      static_entries.append((idx, arg))
  # each call it needs to recapture the static values, so we won't cache it
  custom_vjp = _make_custom_vjp(cls, arg_count, dynamic_indices, static_entries)
  jax_out = custom_vjp(*dynamic_values)
  return _tree_to_torchish(jax_out)


def enable_autograd_function_support():
  """Install the patched torch.autograd.Function bridge."""
  assert _TORCHISH is not None and _TREE_COERCE is not None
  global _PATCH_INSTALLED
  if not _PATCH_INSTALLED:
    Function.apply = classmethod(_patched_apply)
    _PATCH_INSTALLED = True


def disable_autograd_function_support():
  """Revert to PyTorch's original Function.apply implementation."""
  global _PATCH_INSTALLED
  if _PATCH_INSTALLED:
    Function.apply = classmethod(_ORIG_APPLY)
    _PATCH_INSTALLED = False
