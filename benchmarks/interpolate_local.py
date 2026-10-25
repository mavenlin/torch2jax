"""Experimental FP32 local linear interpolation; size-based, align_corners=False, no antialiasing."""

import jax.numpy as jnp


def local_linear(x, shape):
  for axis in range(2, x.ndim):
    source, target = x.shape[axis], shape[axis]
    coord = jnp.maximum((jnp.arange(target, dtype=jnp.float32) + 0.5) * (source / target) - 0.5, 0)
    lo = jnp.floor(coord).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, source - 1)
    weights = (coord - lo).reshape((1,) * axis + (target,) + (1,) * (x.ndim - axis - 1))
    x = jnp.take(x, lo, axis=axis) * (1 - weights) + jnp.take(x, hi, axis=axis) * weights
  return x
