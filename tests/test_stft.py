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
      # GPU overlap accumulation may differ by an FP16 rounding step.
      gradient_rtol = max(4e-5, 2 * max(float(jnp.finfo(x.dtype).eps) for x in args))
      backward_test(lambda *args: torch.abs(call(*args)) ** 2, args, rtol=gradient_rtol, atol=1e-6)


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


@pytest.mark.parametrize("return_complex", [True, False])
def test_stft_vmap_and_composition(return_complex):
  values = jax.random.normal(jax.random.key(3), (3, 64))

  def call(x):
    return torch.stft(x, 16, hop_length=4, return_complex=return_complex)

  converted = t2j(call)
  expected = call(torch.tensor(np.asarray(values))).numpy()
  mapped = jax.jit(jax.vmap(converted))(values)
  np.testing.assert_allclose(mapped, expected, rtol=2e-5, atol=2e-5)
  # Consumers must see the usual logical frequency/time axes.
  reduced = jax.jit(lambda x: jnp.abs(converted(x)).sum(axis=-1))(values)
  np.testing.assert_allclose(reduced, np.abs(expected).sum(axis=-1), rtol=2e-5, atol=2e-5)


def test_stft_bfloat16_input_float32_window():
  values = jax.random.normal(jax.random.key(4), (2, 97)).astype(jnp.bfloat16)
  window = jnp.hanning(17)[:-1]
  source = torch.tensor(np.asarray(values.astype(jnp.float32)), dtype=torch.bfloat16, requires_grad=True)
  torch_window = torch.tensor(np.asarray(window))

  def call(x, w):
    return torch.stft(x, 16, hop_length=4, window=w, return_complex=True)

  expected = call(source, torch_window)
  actual = jax.jit(t2j(call))(values, window)
  np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=2e-5, atol=2e-5)
  expected.abs().square().mean().backward()
  gradient = jax.jit(jax.grad(lambda x: jnp.mean(jnp.abs(t2j(call)(x, window)) ** 2)))(values)
  # BF16 overlap accumulation can round differently across frameworks.
  np.testing.assert_allclose(gradient.astype(jnp.float32), source.grad.float().numpy(), rtol=0.02, atol=0.002)
