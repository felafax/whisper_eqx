import math
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# ruff: noqa: F722

class Linear(eqx.Module):
    bias: Optional[Array]
    weight: Array

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: PRNGKeyArray,
        use_bias = True
    ):
        assert (
            input_dim >= 1 or output_dim >= 1
        ), f"input_dim: {input_dim} | output_dim: {output_dim} are too small"
        wkey, bkey = jax.random.split(key, 2)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        lim = 1 / math.sqrt(input_dim)

        self.weight = jax.random.uniform(
            wkey, (input_dim, output_dim), minval=-lim, maxval=lim
        ) * math.sqrt(1 / (3 * input_dim))

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim)
        else:
            self.bias = jnp.zeros((output_dim,))

    @eqx.filter_jit
    def __call__(
        self,
        arr: Array,
        mask: Optional[Array] = None,
    ) -> Array:

        _mask = jnp.ones_like(self.weight) if mask is None else mask

        return arr @ (self.weight * _mask.astype(arr.dtype)) + self.bias


class FeedForward(eqx.Module):
    fc1: Linear
    fc2: Linear
    dropout: float
    activation: Callable

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        key: PRNGKeyArray,
    ):
        super().__init__()
        key1, key2 = jax.random.split(key)

        self.fc1 = Linear(embed_dim, ff_dim, key=key1)
        self.fc2 = Linear(ff_dim, embed_dim, key=key2)
        self.activation = getattr(jax.nn, activation)
        self.dropout = dropout

        assert self.dropout == 0., 'Non-zero dropout provided.'

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        x = jax.vmap(self.fc1)(x)
        x = self.activation(x)

        x = eqx.nn.Dropout(self.dropout)(x, key=key)
        x = jax.vmap(self.fc2)(x)

        return x

class MultiHeadAttention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    num_heads: int
    head_dim: int
    scale: float

    def __init__(self, embed_dim: int, num_heads: int, is_decoder: bool, *, key):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        keys = jax.random.split(key, 4)
        self.q_proj = Linear(embed_dim, embed_dim, use_bias=True, key=keys[0])
        self.k_proj = Linear(embed_dim, embed_dim, use_bias=is_decoder, key=keys[1])
        self.v_proj = Linear(embed_dim, embed_dim, use_bias=True, key=keys[2])
        self.out_proj = Linear(embed_dim, embed_dim, use_bias=True, key=keys[3])

    def __call__(
        self,
        hidden_states: Float[Array, "s_q d"],
        key_value_states: Optional[Float[Array, "s_kv d"]] = None,
        attention_mask: Optional[Float[Array, "1 s_q s_kv"]] = None,
    ) -> Tuple[Float[Array, "s_q d"], Float[Array, "h s_q s_kv"]]:
        k_input = key_value_states if key_value_states is not None else hidden_states
        q = self.q_proj(hidden_states) * self.scale
        k = self.k_proj(k_input)
        v = self.v_proj(k_input)

        s_q = q.shape[0]  # Query sequence length (decoder side)
        s_kv = k.shape[0]  # Key/Value sequence length (encoder side)

        # Reshape and transpose tensors
        q = q.reshape(s_q, self.num_heads, self.head_dim).transpose(
            1, 0, 2
        )  # [h, s_q, d_head]
        k = k.reshape(s_kv, self.num_heads, self.head_dim).transpose(
            1, 2, 0
        )  # [h, d_head, s_kv]
        v = v.reshape(s_kv, self.num_heads, self.head_dim).transpose(
            1, 0, 2
        )  # [h, s_kv, d_head]

        # Attention scores
        attn_weights = jnp.einsum("hqd,hdk->hqk", q, k)  # [h, s_q, s_kv]

        # Apply attention mask (if provided)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and attention output
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("hqk,hkd->hqd", attn_probs, v)  # [h, s_q, d_head]

        # Merge heads and final projection
        attn_output = attn_output.transpose(1, 0, 2).reshape(s_q, -1)  # [s_q, d]
        output = self.out_proj(attn_output)  # [s_q, d]

        return output, attn_probs
