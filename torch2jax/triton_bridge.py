"""Legacy Triton bridge used by torch2jax."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import numpy as np
import re
import torch
import jax.numpy as jnp
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction, MockTensor

_TRITON_RUNTIME_OPTION_KEYS = {
  "num_warps",
  "num_stages",
  "num_ctas",
  "compute_capability",
  "enable_fp_fusion",
}

_REGISTERED_TORCHISH_TYPES: Tuple[type, ...] = ()
_TORCH_TENSOR_CONVERTER: Optional[Any] = None
_POINTER_PLACEHOLDERS: Dict[str, Tuple[Any, Tuple[int, ...]]] = {}

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
  torchish_types: Iterable[type] = (),
  torch_tensor_converter: Optional[Any] = None,
  pointer_placeholders: Optional[Dict[str, Tuple[Any, Tuple[int, ...]]]] = None,
):
  global _REGISTERED_TORCHISH_TYPES, _TORCH_TENSOR_CONVERTER, _POINTER_PLACEHOLDERS
  _REGISTERED_TORCHISH_TYPES = tuple(torchish_types)
  _TORCH_TENSOR_CONVERTER = torch_tensor_converter
  if pointer_placeholders is not None:
    _POINTER_PLACEHOLDERS = {k: (v[0], tuple(v[1])) for k, v in pointer_placeholders.items()}


def _is_torchish(value: Any) -> bool:
  return any(isinstance(value, cls) for cls in _REGISTERED_TORCHISH_TYPES)


def _as_python_scalar(value: Any) -> Any:
  if hasattr(value, "shape") and getattr(value, "shape") == ():
    try:
      return value.item()
    except Exception:
      return value
  if hasattr(value, "item") and callable(getattr(value, "item")):
    try:
      return value.item()
    except Exception:
      return value
  return value


def _torch_dtype_from_value(value: Any) -> torch.dtype:
  dtype = getattr(value, "dtype", None)
  if isinstance(dtype, torch.dtype):
    return dtype
  if dtype is None:
    return torch.float32
  try:
    np_dtype = np.dtype(dtype)
    tensor = torch.from_numpy(np.zeros((), dtype=np_dtype))
    return tensor.dtype
  except Exception:
    return torch.float32


def _convert_arg(value: Any) -> Any:
  if _is_torchish(value):
    return value.value
  if _TORCH_TENSOR_CONVERTER is not None and isinstance(value, torch.Tensor):
    return _TORCH_TENSOR_CONVERTER(value)
  if isinstance(value, np.ndarray):
    return jnp.asarray(value)
  return value


def _convert_meta(value: Any) -> Any:
  if _is_torchish(value):
    value = value.value
  if hasattr(value, "value"):
    try:
      return value.value
    except Exception:
      pass
  if type(value).__name__ == "constexpr":
    try:
      return value.value
    except Exception:
      pass
  if isinstance(value, jax.Array):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr
  if isinstance(value, jnp.ndarray):
    return value.item() if value.shape == () else np.asarray(value)
  if isinstance(value, np.ndarray):
    return value.item() if value.shape == () else value
  return value


def _is_array_arg(value: Any) -> bool:
  if _is_torchish(value):
    return True
  if isinstance(value, torch.Tensor):
    return True
  if isinstance(value, jax.Array):
    return True
  if isinstance(value, (jnp.ndarray, np.ndarray)):
    return True
  return False


def _placeholder_for(name: str) -> Optional[Any]:
  if name in _POINTER_PLACEHOLDERS:
    dtype, shape = _POINTER_PLACEHOLDERS[name]
    return jnp.zeros(shape, dtype=dtype)
  return None


def _detect_output_indices(
  kernel: Any,
  grid: Any,
  arg_names: Sequence[str],
  array_positions: Dict[int, int],
  pointer_args: Sequence[Any],
  pointer_targets: Sequence[Any],
  call_kwargs: Dict[str, Any],
) -> List[int]:
  if callable(grid) or not array_positions:
    return []

  detection_kwargs = {
    k: _as_python_scalar(v) for k, v in call_kwargs.items()
  }

  detection_args: List[Any] = []
  for idx, name in enumerate(arg_names):
    if idx in array_positions:
      ptr_idx = array_positions[idx]
      source_value = pointer_targets[ptr_idx]
      if source_value is None:
        source_value = pointer_args[ptr_idx]
      detection_args.append(MockTensor(_torch_dtype_from_value(source_value)))
    else:
      candidate = detection_kwargs.get(name, 0)
      detection_args.append(_as_python_scalar(candidate))

  try:
    return _detect_triton_output_indices(
      kernel,
      *detection_args,
      grid=grid,
      **detection_kwargs,
    )
  except Exception:
    return []


def call_triton_with_jax(
  kernel_wrapper: Any,
  grid: Any,
  args: Sequence[Any],
  kwargs: Dict[str, Any],
  *,
  debug: bool = False,
):
  import jax_triton

  kernel, metadata_source = _unwrap_kernel(kernel_wrapper)
  arg_names = list(getattr(metadata_source, "arg_names", ()))
  constexpr_indices = set(getattr(kernel, "constexprs", ()))

  config_defaults: Dict[str, Any] = {}
  if hasattr(kernel_wrapper, "configs") and kernel_wrapper.configs:
    config = getattr(kernel_wrapper, "best_config", None) or kernel_wrapper.configs[0]
    config_defaults.update(config.all_kwargs())

  call_kwargs: Dict[str, Any] = {}
  for key, value in config_defaults.items():
    if key in arg_names or key in _TRITON_RUNTIME_OPTION_KEYS:
      call_kwargs[key] = _convert_meta(value)

  remaining_kwargs = dict(kwargs)
  pointer_args: List[Any] = []
  pointer_targets: List[Any] = []
  array_positions: Dict[int, int] = {}

  for idx, name in enumerate(arg_names):
    value = args[idx] if idx < len(args) else remaining_kwargs.pop(name, None)

    if idx in constexpr_indices:
      meta_source = value if value is not None else config_defaults.get(name)
      if meta_source is not None and name in arg_names:
        call_kwargs[name] = _convert_meta(meta_source)
      continue

    converted = None
    if value is not None:
      converted = _convert_arg(value)
    else:
      placeholder = _placeholder_for(name)
      if placeholder is not None:
        converted = placeholder

    if converted is not None and _is_array_arg(converted):
      pointer_index = len(pointer_args)
      pointer_args.append(converted)
      pointer_targets.append(value)
      array_positions[idx] = pointer_index
      call_kwargs.pop(name, None)
    else:
      meta_source = value if value is not None else config_defaults.get(name)
      if meta_source is not None and name in arg_names:
        call_kwargs[name] = _convert_meta(meta_source)

  for key, value in remaining_kwargs.items():
    if key in arg_names or key in _TRITON_RUNTIME_OPTION_KEYS:
      call_kwargs[key] = _convert_meta(value)

  output_indices = _detect_output_indices(
    kernel,
    grid,
    arg_names,
    array_positions,
    pointer_args,
    pointer_targets,
    call_kwargs,
  )
  if not output_indices and array_positions:
    # Fallback: assume the last array argument is written if IR analysis failed.
    output_indices = [max(array_positions)]

  out_shapes = []
  input_output_aliases = {}
  for out_idx, kernel_index in enumerate(output_indices):
    pointer_index = array_positions.get(kernel_index)
    if pointer_index is None:
      continue
    array_value = pointer_args[pointer_index]
    out_shapes.append(jax.ShapeDtypeStruct(shape=array_value.shape, dtype=array_value.dtype))
    input_output_aliases[pointer_index] = out_idx

  if not out_shapes:
    out_shape = ()
  elif len(out_shapes) == 1:
    out_shape = out_shapes[0]
  else:
    out_shape = tuple(out_shapes)

  if debug:
    call_repr = {k: f"{v!r} ({type(v).__name__})" for k, v in call_kwargs.items()}
    ptr_repr = [f"{type(arg).__name__}" for arg in pointer_args]
    print(
      "[torch2jax] Triton bridge call: "
      f"kernel={getattr(kernel, '__name__', repr(kernel))} "
      f"pointer_args={len(pointer_args)} types={ptr_repr} call_kwargs={call_repr}"
    )

  result = jax_triton.triton_call(
    *pointer_args,
    kernel=kernel,
    out_shape=out_shape,
    grid=grid,
    input_output_aliases=input_output_aliases or None,
    **call_kwargs,
  )

  if output_indices:
    outputs = (result,) if len(output_indices) == 1 else tuple(result)
    for output, kernel_index in zip(outputs, output_indices):
      pointer_index = array_positions.get(kernel_index)
      if pointer_index is None:
        continue
      original = pointer_targets[pointer_index]
      if hasattr(original, "value"):
        original.value = output
  return None


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
  for match in re.finditer(r"%arg(\d+):\s*([^,)]+)", header):
    idx = int(match.group(1))
    ty = match.group(2).strip()
    if "!tt.ptr" in ty:
      pointer_params[f"%arg{idx}"] = idx

  pointer_base = dict(pointer_params)
  outputs: set[int] = set()

  assign_re = re.compile(r"%(?P<dest>[A-Za-z0-9_]+)\s*=\s*(?P<op>[A-Za-z0-9_.]+)\s*(?P<rest>.*)")
  ssa_token_re = re.compile(r"%[A-Za-z0-9_]+")
  pointer_ops_without_arrow = {"tt.addptr", "tt.make_tensor_ptr", "tt.make_ptr"}

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
