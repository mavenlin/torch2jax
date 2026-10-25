"""PyTorch-oracle interpolation tests, including nonuniform vector-Jacobian products.

Run on CPU or CUDA with JAX_ENABLE_X64=true. CUDA is selected when JAX uses a GPU.
The exhaustive option matrix is FP32; additional matrices exercise FP64, FP16,
BF16, structured signals, noncontiguous inputs, and the experimental local kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch2jax import t2j

MODE_RANKS = [(mode, rank) for mode in ("nearest", "nearest-exact", "area") for rank in (1, 2, 3)] + [
  ("linear", 1),
  ("bilinear", 2),
  ("trilinear", 3),
  ("bicubic", 2),
]
LINEAR_MODES = ("linear", "bilinear", "trilinear", "bicubic")
# Fixed before executing the suite. BF16 has a 7-bit mantissa; its forward and
# gradient reductions can round at different stages in the two implementations.
TOLERANCES = {
  "float64": (2e-10, 2e-10, 2e-10),
  "float32": (1e-5, 2e-5, 3e-6),
  "float16": (3e-3, 4e-3, 3e-3),
  "bfloat16": (3e-2, 3e-2, 2e-2),
}


def settings(rank):
  spatial = (5, 7, 3)[:rank]
  return [
    ("identity", spatial, {"size": spatial}),
    ("up", spatial, {"size": tuple(s * 2 for s in spatial)}),
    ("down", spatial, {"size": tuple(max(1, s // 2) for s in spatial)}),
    ("mixed", spatial, {"size": (3, 11, 2)[:rank]}),
    ("scalar_size", spatial, {"size": 4}),
    ("singleton_out", spatial, {"size": (1,) * rank}),
    ("singleton_in", (1,) + spatial[1:], {"size": (4, 3, 6)[:rank]}),
    ("axis_singleton", spatial, {"size": (1, 9, 1)[:rank]}),
    ("scale_identity", spatial, {"scale_factor": 1.0}),
    ("scale_integer", spatial, {"scale_factor": 2.0}),
    ("scale_down", spatial, {"scale_factor": 0.7}),
    ("scale_up", spatial, {"scale_factor": 1.7}),
    ("scale_tuple", spatial, {"scale_factor": (0.7, 1.7, 2.3)[:rank]}),
    ("scale_recompute", spatial, {"scale_factor": 1.7, "recompute_scale_factor": True}),
    ("scale_no_recompute", spatial, {"scale_factor": 1.7, "recompute_scale_factor": False}),
    ("tuple_recompute", spatial, {"scale_factor": (0.7, 1.7, 2.3)[:rank], "recompute_scale_factor": True}),
  ]


def cases(full):
  result = []
  for mode, rank in MODE_RANKS:
    for name, spatial, options in settings(rank):
      if not full and name not in ("mixed", "singleton_out", "singleton_in", "scale_tuple", "scale_recompute"):
        continue
      corners = (None, False, True) if mode in LINEAR_MODES else (None,)
      for align in corners:
        for aa in (False, True) if mode in ("bilinear", "bicubic") else (False,):
          kwargs = dict(mode=mode, **options)
          if mode in LINEAR_MODES:
            kwargs["align_corners"] = align
          kwargs["antialias"] = aa
          result.append(pytest.param((2, 2) + spatial, kwargs, id=f"{mode}-{rank}d-{name}-align{align}-aa{aa}"))
  return result


@pytest.fixture(scope="module", autouse=True)
def enable_x64():
  with jax.enable_x64():
    yield


@pytest.fixture(autouse=True)
def bound_compilation_cache():
  # The suite compiles hundreds of different shapes. Keep memory bounded on CI.
  yield
  bound_compilation_cache.count += 1
  if bound_compilation_cache.count % 40 == 0:
    jax.clear_caches()


bound_compilation_cache.count = 0


def numpy_value(value):
  if isinstance(value, torch.Tensor):
    return value.detach().to(dtype=torch.float64).cpu().numpy()
  return np.asarray(value.astype(jnp.float64))


def assert_close(actual, expected, dtype, *, gradient=False, exact=False, tolerance=None):
  a, b = numpy_value(actual), numpy_value(expected)
  assert a.shape == b.shape
  assert np.isfinite(a).all(), "JAX returned nonfinite values for finite test data"
  if exact:
    np.testing.assert_array_equal(a, b)
    return
  atol, rtol, norm_tolerance = TOLERANCES[dtype] if tolerance is None else tolerance
  np.testing.assert_allclose(a, b, atol=atol * (2 if gradient else 1), rtol=rtol)
  # Normalize by at least a unit RMS signal: structured tests can produce exact
  # zero through cancellation, where a purely relative norm is undefined.
  relative_l2 = np.linalg.norm(a - b) / max(np.linalg.norm(b), np.sqrt(b.size))
  assert relative_l2 <= norm_tolerance, f"relative L2 error {relative_l2} > {norm_tolerance}"


def check_parity(
  shape, kwargs, dtype="float32", pattern="random", seed=123, noncontiguous=False, kernel=None, tolerance=None
):
  rng = np.random.default_rng(seed)
  data = rng.normal(size=shape)
  if pattern == "constant":
    data.fill(0.75)
  elif pattern == "ramp":
    data = np.broadcast_to(np.linspace(-1, 1, shape[-1]), shape).copy()
  elif pattern == "impulse":
    data.fill(0)
    data[(slice(None), slice(None)) + (0,) * (len(shape) - 2)] = 1
  elif pattern == "checkerboard":
    spatial = np.indices(shape[2:]).sum(axis=0) % 2
    data = np.broadcast_to(2 * spatial - 1, shape).copy()
  device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
  x = torch.tensor(data, dtype=getattr(torch, dtype), device=device)
  if noncontiguous:
    storage = torch.empty(shape[:-1] + (shape[-1] * 2,), dtype=x.dtype, device=device)
    storage[..., ::2] = x
    x = storage[..., ::2]
    assert not x.is_contiguous()
  x.requires_grad_(True)
  # Quantize once in Torch, then pass exactly the same input values to JAX.
  jx = jnp.asarray(numpy_value(x), dtype=getattr(jnp, dtype))
  try:
    reference = F.interpolate(x, **kwargs)
  except NotImplementedError as exc:
    if (
      device == "cpu"
      and dtype in ("float16", "bfloat16")
      and kwargs.get("antialias")
      and "compute_index_ranges_weights" in str(exc)
    ):
      pytest.skip(f"PyTorch CPU does not implement antialias for {dtype}; covered on CUDA")
    raise
  cotangent = torch.tensor(rng.normal(size=tuple(reference.shape)), device=device, dtype=x.dtype)
  gradient = torch.autograd.grad(reference, x, cotangent)[0]
  jcotangent = jnp.asarray(numpy_value(cotangent), dtype=jx.dtype)
  fn = t2j(partial(F.interpolate, **kwargs)) if kernel is None else kernel

  def value_and_vjp(value, cot):
    out, pullback = jax.vjp(fn, value)
    return out, pullback(cot)[0]

  results = [evaluate(jx, jcotangent) for evaluate in (value_and_vjp, jax.jit(value_and_vjp))]
  for out, grad in results:
    assert out.dtype == jx.dtype, f"output dtype changed: {out.dtype} != {jx.dtype}"
    assert grad.dtype == jx.dtype
    assert_close(out, reference, dtype, exact=kwargs["mode"] in ("nearest", "nearest-exact"), tolerance=tolerance)
  # PyTorch CPU nearest3d backward uses a legacy 2x shortcut even when a
  # fractional scale factor produced that output size. Its forward does not.
  # Validate JAX against the exact transpose of Torch's observed sampling map
  # before recording this narrowly scoped upstream-oracle discrepancy.
  if (
    device == "cpu"
    and kwargs["mode"] == "nearest"
    and len(shape) == 5
    and kwargs.get("scale_factor") == (0.7, 1.7, 2.3)
    and not kwargs.get("recompute_scale_factor")
  ):
    labels = torch.arange(x.numel(), dtype=torch.float64).reshape(shape)
    indices = F.interpolate(labels, **kwargs).numpy().astype(np.intp).ravel()
    exact_grad = np.zeros(x.numel(), dtype=np.float64)
    np.add.at(exact_grad, indices, numpy_value(cotangent).ravel())
    exact_grad = exact_grad.reshape(shape)
    for _, grad in results:
      assert_close(grad, jnp.asarray(exact_grad, dtype=jx.dtype), dtype, gradient=True)
    if not np.allclose(numpy_value(gradient), exact_grad, atol=TOLERANCES[dtype][0] * 2, rtol=TOLERANCES[dtype][1]):
      pytest.xfail(
        "PyTorch CPU nearest3d backward disagrees with its forward sampling map for scale_factor=(0.7,1.7,2.3); JAX matches the exact transpose"
      )
  for _, grad in results:
    assert_close(grad, gradient, dtype, gradient=True, tolerance=tolerance)


@pytest.mark.parametrize("shape,kwargs", cases(full=True))
def test_interpolate_fp32_options(shape, kwargs):
  check_parity(shape, kwargs)


@pytest.mark.parametrize("dtype", ["float64", "float16", "bfloat16"])
@pytest.mark.parametrize("shape,kwargs", cases(full=False))
def test_interpolate_dtypes(shape, kwargs, dtype):
  check_parity(shape, kwargs, dtype)


@pytest.mark.parametrize("mode,rank", MODE_RANKS)
@pytest.mark.parametrize("pattern", ["constant", "ramp", "impulse", "checkerboard"])
def test_interpolate_structured_signals(mode, rank, pattern):
  kwargs = dict(mode=mode, size=(9, 3, 6)[:rank])
  if mode in LINEAR_MODES:
    kwargs["align_corners"] = False
  check_parity((1, 2) + (5, 7, 3)[:rank], kwargs, pattern=pattern)


@pytest.mark.parametrize("mode,rank", MODE_RANKS)
def test_interpolate_noncontiguous(mode, rank):
  check_parity((2, 3) + (5, 7, 3)[:rank], dict(mode=mode, size=(3, 9, 2)[:rank]), noncontiguous=True)


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
@pytest.mark.parametrize("source,target", [(2, 3), (3, 2), (7, 10), (10, 7), (17, 29), (257, 513)])
def test_interpolate_nearest_boundaries(mode, source, target):
  check_parity((1, 2, source), dict(mode=mode, size=target), pattern="ramp")


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_interpolate_antialias_boundaries(seed, mode):
  check_parity((1, 3, 9, 11), dict(mode=mode, size=(2, 3), align_corners=True, antialias=True), seed=seed)


@pytest.mark.parametrize("mode,rank", [("linear", 1), ("bilinear", 2), ("trilinear", 3)])
@pytest.mark.parametrize("pattern", ["random", "constant", "ramp", "impulse", "checkerboard"])
@pytest.mark.parametrize(
  "configuration", ["identity", "up", "down", "mixed", "singleton_out", "singleton_in", "axis_singleton"]
)
def test_interpolate_local_prototype(mode, rank, pattern, configuration):
  # This prototype currently accepts only a target size and FP32 input. Do not
  # imply coverage of align_corners, antialiasing, scale_factor, or other modes.
  from benchmarks.interpolate_local import local_linear

  _, spatial, kwargs = next(case for case in settings(rank) if case[0] == configuration)
  shape = (1, 2) + spatial
  target = shape[:2] + kwargs["size"]
  check_parity(
    shape, dict(mode=mode, align_corners=False, **kwargs), pattern=pattern, kernel=partial(local_linear, shape=target)
  )


def test_nearest_fractional_vjp_matches_finite_difference():
  # Minimal reproducer for the CPU oracle issue: forward samples [0,0,0,1,1,2],
  # so cotangents [1,2,3,4,5,6] must give [6,9,6], not [3,7,11].
  x = torch.arange(3.0, dtype=torch.float64).reshape(1, 1, 1, 1, 3)
  cot = torch.arange(1.0, 7.0, dtype=torch.float64).reshape(1, 1, 1, 1, 6)
  fn = partial(F.interpolate, mode="nearest", scale_factor=(1, 1, 2.3))
  eps = 1e-5
  finite_difference = []
  for i in range(3):
    delta = torch.zeros_like(x)
    delta.flatten()[i] = eps
    finite_difference.append(((fn(x + delta) - fn(x - delta)) * cot).sum().item() / (2 * eps))
  jx = jnp.asarray(x.numpy())
  _, pullback = jax.vjp(t2j(fn), jx)
  np.testing.assert_allclose(
    np.asarray(pullback(jnp.asarray(cot.numpy()))[0]).ravel(), finite_difference, atol=1e-8, rtol=1e-8
  )


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "bfloat16"])
@pytest.mark.parametrize(
  "shape,kwargs",
  [
    ((1, 2, 129, 131), dict(mode="bilinear", size=(193, 197), align_corners=False)),
    ((1, 2, 129, 131), dict(mode="bilinear", scale_factor=1.7, align_corners=False)),
    ((1, 2, 129, 131), dict(mode="bilinear", size=(33, 29), align_corners=True, antialias=True)),
    ((1, 2, 129, 131), dict(mode="bicubic", size=(193, 197), align_corners=False)),
    ((1, 2, 129, 131), dict(mode="bicubic", size=(33, 29), align_corners=True, antialias=True)),
    ((1, 2, 9, 17, 19), dict(mode="trilinear", size=(15, 23, 31), align_corners=False)),
  ],
)
def test_interpolate_larger_fractional_grids(shape, kwargs, dtype):
  # FP32 source-coordinate rounding grows with image size. At 129-131 pixels,
  # reciprocal/FMA rounding can move a sample by about 1e-5 pixels. Keep the
  # small-grid budget tight, and bound these cases by 1e-4 pointwise / 1e-5 RMS.
  # The same grids must still satisfy the original FP64 / FP16 / BF16 budgets.
  tolerance = (1e-4, 2e-5, 1e-5) if dtype == "float32" else None
  check_parity(shape, kwargs, dtype, tolerance=tolerance)


def test_interpolate_local_prototype_larger_fractional_grid():
  from benchmarks.interpolate_local import local_linear

  shape = (1, 2, 129, 131)
  target = (1, 2, 193, 197)
  check_parity(
    shape, dict(mode="bilinear", size=target[2:], align_corners=False), kernel=partial(local_linear, shape=target)
  )


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "bfloat16"])
@pytest.mark.parametrize("mode,rank", MODE_RANKS)
def test_interpolate_small_gradients_relative_accuracy(mode, rank, dtype):
  # Absolute tolerances must not let a missing or incorrectly scaled gradient
  # pass when downsampling spreads a small cotangent over many source pixels.
  device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
  rng = np.random.default_rng(42)
  shape = (2, 2) + (5, 7, 3)[:rank]
  x = torch.tensor(rng.normal(size=shape), dtype=getattr(torch, dtype), device=device, requires_grad=True)
  fn = partial(F.interpolate, mode=mode, size=1)
  out = fn(x)
  cot = torch.tensor(rng.uniform(0.1, 0.2, size=tuple(out.shape)), dtype=x.dtype, device=device)
  expected = numpy_value(torch.autograd.grad(out, x, cot)[0])
  jx = jnp.asarray(numpy_value(x), dtype=getattr(jnp, dtype))
  jc = jnp.asarray(numpy_value(cot), dtype=jx.dtype)

  def vjp(value, weights):
    _, pullback = jax.vjp(t2j(fn), value)
    return pullback(weights)[0]

  for evaluate in (vjp, jax.jit(vjp)):
    actual = numpy_value(evaluate(jx, jc))
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error <= TOLERANCES[dtype][2], f"small-gradient relative error: {relative_error}"
