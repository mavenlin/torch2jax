import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from torch2jax import t2j

from .utils import backward_test, forward_test


@pytest.mark.parametrize(
  "shape,options,dtype,window_dtype,member",
  [
    ((97,), {}, jnp.float32, None, False),
    ((2, 97), dict(win_length=13, hop_length=5), jnp.float32, jnp.float32, False),
    ((2, 97), dict(win_length=16, center=False, align_to_window=False), jnp.float32, jnp.float32, False),
    ((97,), dict(win_length=16, normalized=True, onesided=False), jnp.float64, jnp.float32, False),
    ((2, 97), dict(win_length=13, return_complex=False), jnp.float32, jnp.float64, False),
    ((97,), dict(n_fft=15, win_length=10, pad_mode="constant"), jnp.float32, jnp.float32, False),
    ((97,), dict(win_length=13, pad_mode="replicate", return_complex=None), jnp.complex64, jnp.float32, True),
    ((97,), dict(win_length=13, pad_mode="circular", return_complex=None), jnp.float64, jnp.complex64, False),
    ((97,), dict(win_length=14, center=False, align_to_window=True), jnp.complex128, jnp.complex128, False),
    ((2, 98), dict(win_length=13, center=False, align_to_window=True), jnp.float32, jnp.float32, False),
    ((97,), dict(win_length=13), jnp.float32, jnp.int32, False),
    ((97,), dict(win_length=13), jnp.float32, jnp.bool_, False),
    ((97,), dict(win_length=13), jnp.float16, jnp.float32, False),
  ],
)
def test_stft_forward_and_gradient(shape, options, dtype, window_dtype, member):
  options = dict(n_fft=16, return_complex=True) | options

  def call(x, window=None):
    return x.stft(window=window, **options) if member else torch.stft(x, window=window, **options)

  with jax.enable_x64():
    args = [jax.random.normal(jax.random.key(0), shape, dtype=dtype)]
    if window_dtype is not None:
      sample_dtype = window_dtype if jnp.issubdtype(window_dtype, jnp.inexact) else jnp.float32
      window = jax.random.normal(jax.random.key(1), (options["win_length"],), dtype=sample_dtype)
      args.append(window > 0 if window_dtype == jnp.bool_ else window.astype(window_dtype))
    tolerance = 1e-11 if jnp.result_type(*args) in (jnp.float64, jnp.complex128) else 1e-5
    forward_test(call, args, rtol=3 * tolerance, atol=tolerance)
    if all(jnp.issubdtype(x.dtype, jnp.floating) for x in args):
      backward_test(lambda *args: torch.abs(call(*args)) ** 2, args, rtol=4e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64, jnp.complex128])
def test_stft_jitted_gradient(dtype):
  with jax.enable_x64():
    values = jax.random.normal(jax.random.key(0), (32,), dtype=dtype)
    source = torch.tensor(np.asarray(values), requires_grad=True)
    power = lambda x: torch.abs(torch.stft(x, 8, return_complex=True)) ** 2
    power(source).mean().backward()
    gradient = jax.jit(jax.grad(lambda x: t2j(power)(x).mean()))(values)
    # Torch and JAX use conjugate conventions for a real loss of complex inputs.
    np.testing.assert_allclose(jnp.conj(gradient), source.grad.numpy(), rtol=4e-5, atol=1e-6)


def test_stft_rejects_incomplete_aligned_frame():
  options = dict(n_fft=16, win_length=13, hop_length=4, center=False, align_to_window=True, return_complex=True)
  with pytest.raises(RuntimeError):
    torch.stft(torch.ones(33), **options)
  with pytest.raises(RuntimeError, match="aligned FFT frame extends past the padded input"):
    t2j(lambda x: torch.stft(x, **options))(jnp.ones(33))
