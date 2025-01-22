import math
import jax.numpy as jnp
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

def causal_mask(seq_len: int) -> Float[Array, "1 1 s s"]:
    return jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))
