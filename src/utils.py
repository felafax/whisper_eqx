import math
from typing import Callable

import jax.numpy as jnp
import numpy as np
import termplotlib as tpl
from jaxtyping import Array, Float, Int
from torch import Tensor

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

def causal_mask(seq_len: int, padding_mask: Array) -> Float[Array, "1 1 s t"]:
    """
    Create a causal mask from provided `seq_len` and also incorporate `padding_mask`
    to prevent attention going to padding tokens.
    """
    min_dtype = jnp.finfo(jnp.float32).min
    padding_mask = jnp.ones(padding_mask.shape[0])[:, None] @ padding_mask[:, None].T
    padding_mask = jnp.broadcast_to(padding_mask, (1, 1, seq_len, seq_len))

    causal_mask = jnp.triu(jnp.ones((1, 1, seq_len, seq_len)), k=1)
    full_mask = jnp.where(padding_mask * (1 - causal_mask) == 1, -0.0, min_dtype)

    return full_mask

def ascii_hist(x: Array, bins: int = 10):
    n, xedges = np.histogram(x, bins=bins)
    max_n = n.max()
    bar_width = 50
    if max_n > 0:
        normed_n = n / max_n
    else:
        normed_n = n  # Avoid division by zero if max_n is zero

    for i in range(len(n)):
        bar = "#" * int(normed_n[i] * bar_width)
        bin_center = (xedges[i] + xedges[i + 1]) / 2
        bin_str = "{0: <8.4g}".format(bin_center).ljust(10)
        count_str = f"({n[i]})"
        print(f"{bin_str}| {bar} {count_str}")
        
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
    
def parse_path(path_str: str):
    parts = []
    
    for component in path_str.split("."):
        if "[" in component and component.endswith("]"):
            attr_part, index_part = component.split("[")
            index = int(index_part[:-1])
            parts.append(attr_part)
            parts.append(index)
        else:
            parts.append(component)
            
    return parts

def get_hf_param(hf_model, path_str: str) -> Tensor:
    parts = parse_path(path_str)
    current = hf_model
    
    for part in parts:
        if isinstance(part, int):
            current = current[part]
        else:
            current = getattr(current, part)
            
    return current

def create_where_func(path_str: str) -> Callable:
    parts = parse_path(path_str)

    def where_func(model):
        current = model
        for part in parts:
            if isinstance(part, int):
                current = current[part]
            else:
                current = getattr(current, part)
        return current

    return where_func
