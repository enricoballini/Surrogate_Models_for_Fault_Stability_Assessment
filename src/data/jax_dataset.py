"""Benchmark data for JAX users."""

import jax
import jax.numpy as jnp

from src.data.dataset import load_samples


def load_jax_samples(split):
    """Samples of "train" or "validation" as JAX arrays.

    inputs: (n_samples, 10), the 9 parameters followed by the time
    cff:    (n_samples, 416), the CFF on the fault cells, in MPa
    """
    inputs, cff = load_samples(split)
    return jnp.asarray(inputs), jnp.asarray(cff)


def minibatches(inputs, cff, batch_size, key):
    """Go once through the samples in random order, batch_size samples at a time."""
    order = jax.random.permutation(key, len(inputs))
    for start in range(0, len(inputs), batch_size):
        batch = order[start : start + batch_size]
        yield inputs[batch], cff[batch]
