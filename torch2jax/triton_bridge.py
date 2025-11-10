"""Legacy Triton bridge used by torch2jax."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import numpy as np
import re
import torch
import jax.numpy as jnp
import jax_triton
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction, MockTensor

_TRITON_RUNTIME_OPTION_KEYS = {
  "num_warps",
  "num_stages",
  "num_ctas",
  "compute_capability",
  "enable_fp_fusion",
}

Torchish: type = None
j2t_dtype: Callable = None


def _unwrap_kernel(kernel_wrapper: Any) -> Tuple[Any, Any]:
  kernel = kernel_wrapper
  metadata = kernel_wrapper if hasattr(kernel_wrapper, "arg_names") else None

  while True:
    next_candidate = getattr(kernel, "fn", None)
    if next_candidate is None:
      break
    if hasattr(next_candidate, "arg_names"):
      kernel = next_candidate
      metadata = kernel
      continue
    break

  if metadata is None:
    metadata = kernel
  return kernel, metadata


def configure_torchish_support(
  *,
  _torchish: type,
  _j2t_dtype: Callable,
):
  global Torchish, j2t_dtype
  Torchish = _torchish
  j2t_dtype = _j2t_dtype


def _detect_output_indices(
  kernel: Any,
  grid: Any,
  args: Sequence[Any],
  kwargs: Dict[str, Any],
  array_args_indices: List[int],
) -> List[int]:
  args = [MockTensor(j2t_dtype(arg.dtype)) if isinstance(arg, jax.Array) else arg
          for arg in args]
  indice = _detect_triton_output_indices(
    kernel,
    *args,
    grid=grid,
    **kwargs,
  )
  output_indice = [array_args_indices[i] for i in indice]
  input_output_aliases = {i: o for o, i in enumerate(indice)}
  return output_indice, input_output_aliases


def call_triton_with_jax(
  kernel_wrapper: Any,
  grid: Any,
  args: Sequence[Any],
  kwargs: Dict[str, Any],
):
  kernel, metadata_source = _unwrap_kernel(kernel_wrapper)
  arg_names = list(getattr(metadata_source, "arg_names", ()))
  constexpr_indices = set(getattr(kernel, "constexprs", ()))

  config_defaults: Dict[str, Any] = {}
  if hasattr(kernel_wrapper, "configs") and kernel_wrapper.configs:
    config = getattr(kernel_wrapper, "best_config", None) or kernel_wrapper.configs[0]
    config_defaults.update(config.all_kwargs())

  remaining_kwargs = dict(kwargs)

  torchish_args = []
  call_args = []
  detect_args = []
  call_kwargs = {}
  array_args_indices = []
  min_constexpr = min(constexpr_indices)
  for idx, name in enumerate(arg_names):
    arg = args[idx] if idx < len(args) else remaining_kwargs.pop(name, None)
    if idx >= min_constexpr: # anything beyond this point should be passed as kwargs
      meta_arg = arg if arg is not None else config_defaults.get(name)
      call_kwargs[name] = meta_arg
    elif isinstance(arg, Torchish):
      call_args.append(arg.value)
      detect_args.append(arg.value)
      torchish_args.append(arg)
      array_args_indices.append(idx)
    elif arg is None:
      placeholder = jnp.zeros((1,), dtype=jnp.float32)
      call_args.append(placeholder)
      detect_args.append(placeholder)
      torchish_args.append(arg)
      array_args_indices.append(idx)
    elif isinstance(arg, torch.Tensor):
      raise ValueError(f"{name} passed to {kernel} is a torch.Tensor")
    else:
      detect_args.append(arg)
      if isinstance(arg, float):
        arg = np.float32(arg)
      call_args.append(arg)
  # dump all remaining kwargs into the function
  # call_kwargs.update(remaining_kwargs)
  output_indices, input_output_aliases = _detect_output_indices(
    kernel,
    grid,
    detect_args,
    call_kwargs,
    array_args_indices,
  )
  out_shapes = jax.tree.map(lambda i: jax.ShapeDtypeStruct(call_args[i].shape, call_args[i].dtype), output_indices)
  result = jax_triton.triton_call(
    *call_args,
    kernel=kernel,
    out_shape=out_shapes,
    grid=grid,
    input_output_aliases=input_output_aliases,
    **call_kwargs,
  )
  for i, o in input_output_aliases.items():
    torchish_args[i].value = result[o]

def _unwrap_jit_function_for_ir(kernel: Any) -> JITFunction:
  current = kernel
  visited = set()
  while not isinstance(current, JITFunction):
    obj_id = id(current)
    if obj_id in visited:
      raise ValueError("Encountered cycle while unwrapping Triton kernel wrappers.")
    visited.add(obj_id)

    if isinstance(current, Heuristics):
      current = current.fn
      continue

    if isinstance(current, Autotuner):
      current = current.fn
      continue

    if hasattr(current, "fn"):
      current = current.fn
      continue

    if hasattr(current, "base_fn"):
      current = current.base_fn
      continue

    raise ValueError(f"Unable to unwrap Triton kernel to a JITFunction (type={type(current)!r}).")

  return current


def _scan_pointer_outputs(ttir: str) -> List[int]:
  header_match = re.search(r"tt\.func public @\w+\((.*?)\)\s*attributes", ttir, re.DOTALL)
  if not header_match:
    raise ValueError("Could not locate tt.func header in TTIR.")

  header = header_match.group(1)
  pointer_params: Dict[str, int] = {}
  arg_pattern = re.compile(r"%(?P<name>[A-Za-z0-9_]+):\s*(?P<type>[^,)]+)")
  for idx, match in enumerate(arg_pattern.finditer(header)):
    name = match.group("name")
    ty = match.group("type").strip()
    if "!tt.ptr" in ty or "!ttg.ptr" in ty:
      token = f"%{name}"
      pointer_params[token] = idx
      # Newer Triton uses argument names (e.g. %q) while older
      # versions used %arg0. Preserve the legacy alias so either
      # form can participate in pointer propagation.
      if name.startswith("arg"):
        suffix = name[3:]
        if suffix.isdigit():
          pointer_params[f"%arg{int(suffix)}"] = idx

  pointer_base = dict(pointer_params)
  outputs: set[int] = set()

  assign_re = re.compile(r"%(?P<dest>[A-Za-z0-9_]+)\s*=\s*(?P<op>[A-Za-z0-9_.]+)\s*(?P<rest>.*)")
  ssa_token_re = re.compile(r"%[A-Za-z0-9_]+")
  pointer_ops_without_arrow = {
    "tt.addptr",
    "tt.make_tensor_ptr",
    "tt.make_ptr",
    "ttg.addptr",
    "ttg.make_tensor_ptr",
  }

  for raw_line in ttir.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue

    if line.startswith("tt.store"):
      tokens = ssa_token_re.findall(line)
      if tokens:
        ptr_token = tokens[0]
        arg_idx = pointer_base.get(ptr_token)
        if arg_idx is not None:
          outputs.add(arg_idx)
      continue

    match = assign_re.match(line)
    if not match:
      continue

    dest = f"%{match.group('dest')}"
    op = match.group("op")
    rest = match.group("rest")

    propagated = False
    if "->" in rest:
      result_part = rest.split("->", 1)[1]
      if "!tt.ptr" in result_part:
        for token in ssa_token_re.findall(rest):
          if token == dest:
            continue
          if token in pointer_base:
            pointer_base[dest] = pointer_base[token]
            propagated = True
            break

    if not propagated and op in pointer_ops_without_arrow:
      for token in ssa_token_re.findall(rest):
        if token == dest:
          continue
        if token in pointer_base:
          pointer_base[dest] = pointer_base[token]
          propagated = True
          break

    if not propagated and op in {"tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.splat"}:
      for token in ssa_token_re.findall(rest):
        if token == dest:
          continue
        if token in pointer_base:
          pointer_base[dest] = pointer_base[token]
          break

  return sorted(outputs)


def _detect_triton_output_indices(kernel, *args, grid, **meta) -> List[int]:
  jit_fn = _unwrap_jit_function_for_ir(kernel)
  compiled_kernel = jit_fn.warmup(*args, grid=grid, **meta)
  return _scan_pointer_outputs(compiled_kernel.asm["ttir"])
