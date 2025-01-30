import math
import numpy as np
import jax.numpy as jnp
import termplotlib as tpl
from jaxtyping import Array, Float, Int

# ruff: noqa: F722


def sinusoids(length: int, channels: int) -> Float[Array, "length channels"]:
    """Sinusoidal positional embeddings."""
    if channels % 2 != 0:
        raise ValueError(f"Channels must be even, got {channels}")
    log_timescale = math.log(10000.0) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length)[:, None] * inv_timescales[None, :]
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)


def shift_tokens_right(
    input_ids: Int[Array, "b s"], pad_token_id: int, decoder_start_token_id: int
) -> Int[Array, "b s"]:
    shifted = jnp.roll(input_ids, 1, axis=1)
    shifted = shifted.at[:, 0].set(decoder_start_token_id)
    shifted = jnp.where(shifted == -100, pad_token_id, shifted)
    return shifted


def causal_mask(seq_len: int) -> Float[Array, "1 1 s t"]:
    return (
        jnp.triu(jnp.ones((1, 1, seq_len, seq_len)), k=1) * jnp.finfo(jnp.float32).min
    )


def plot_deviation_histogram(hf_output, eqx_output, bins=20):
    # Calculate absolute differences
    diff = jnp.abs(eqx_output - hf_output)
    diff_flat = diff.ravel()

    # Create histogram data
    counts, bin_edges = np.histogram(diff_flat, bins=bins)

    # Create the figure
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=True)

    print("\nDistribution of Deviations between HuggingFace and Equinox outputs")
    print("Frequency →")
    print("↓ Absolute Deviation")
    fig.show()

    # Print summary statistics
    print("\nSummary Statistics:\n")
    print(f"- Min deviation: {np.min(diff_flat):.2e}")
    print(f"- Max deviation: {np.max(diff_flat):.2e}")
    print(f"- Mean deviation: {np.mean(diff_flat):.2e}")
    print(f"- Median deviation: {np.median(diff_flat):.2e}")
