import math
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from layers import FeedForward, MultiHeadAttention
from utils import sinusoids

# ruff: noqa: F722

class SinusoidalPositionalEmbedding(eqx.Module):
    weight: Array

    def __init__(self, num_positions: int, embedding_dim: int, *, key=None):
        super().__init__()
        self.weight = self.sinusoids(num_positions, embedding_dim)

    @staticmethod
    def sinusoids(length: int, channels: int) -> Array:
        if channels % 2 != 0:
            raise ValueError(f"Channels must be even, got {channels}")
        log_timescale = math.log(10000.0) / (channels // 2 - 1)
        inv_timescales = jnp.exp(-log_timescale * jnp.arange(channels // 2))
        scaled_time = jnp.arange(length)[:, None] * inv_timescales[None, :]
        return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)


class EncoderLayer(eqx.Module):
    self_attn: MultiHeadAttention
    self_attn_layer_norm: eqx.nn.LayerNorm
    ff: FeedForward
    final_layer_norm: eqx.nn.LayerNorm
    dropout: float

    def __init__(self, config, *, key):
        super().__init__()
        key1, key2 = jax.random.split(key)

        self.self_attn = MultiHeadAttention(
            config.d_model, config.encoder_attention_heads, False, key=key1
        )
        self.self_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.ff = FeedForward(
            config.d_model,
            config.encoder_ffn_dim,
            config.activation_function,
            config.activation_dropout,
            key=key2,
        )
        self.final_layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def __call__(
        self,
        x: Float[Array, "s d"],
        attn_mask: Optional[Float[Array, "1 s s"]],
        *,
        key,
    ) -> Float[Array, "s d"]:
        keys = jax.random.split(key, 2)

        residual = x
        x = jax.vmap(self.self_attn_layer_norm)(x)
        x, _ = self.self_attn(x, attention_mask=attn_mask)
        x = eqx.nn.Dropout(self.dropout)(x, key=keys[0])
        x = residual + x

        residual = x
        x = jax.vmap(self.final_layer_norm)(x)
        x = self.ff(x, key=keys[1])
        x = residual + x

        return x


class WhisperEncoder(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    embed_positions: eqx.nn.Embedding
    layers: list[EncoderLayer]
    layer_norm: eqx.nn.LayerNorm
    dropout: float

    def __init__(self, config, *, key: PRNGKeyArray):
        super().__init__()
        keys = jax.random.split(key, 4)

        # Convolutional subsampler - input shape [channels, time]
        self.conv1 = eqx.nn.Conv1d(
            in_channels=config.num_mel_bins,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1,
            key=keys[0],
        )

        self.conv2 = eqx.nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            key=keys[1],
        )

        # Positional embeddings (learnable)
        self.embed_positions = eqx.nn.Embedding(
            config.max_source_positions, config.d_model, key=keys[2]
        )

        sinusoid = sinusoids(config.max_source_positions, config.d_model)

        self.embed_positions = eqx.tree_at(
            lambda e: e.weight, self.embed_positions, sinusoid
        )

        # Transformer layers
        self.layers = [
            EncoderLayer(config, key=k)
            for k in jax.random.split(keys[3], config.encoder_layers)
        ]

        self.layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def __call__(
        self,
        input_features: Float[Array, "c t"],  # [channels, time]
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "t d"]:
        key1, key2 = jax.random.split(key)

        x = jax.nn.gelu(self.conv1(input_features))  # [time, d_model]
        x = jax.nn.gelu(self.conv2(x))  # [time//2, d_model]

        # Add positional embeddings
        positions = jax.vmap(self.embed_positions)(
            jnp.arange(x.shape[1], dtype=jnp.int32)
        )

        x = x + positions.T
        x: Array = eqx.nn.Dropout(self.dropout)(x, key=key1).T

        for layer in self.layers:
            x = layer(x, attn_mask=None, key=key2)

        return jax.vmap(self.layer_norm)(x)


class DecoderLayer(eqx.Module):
    self_attn: MultiHeadAttention
    self_attn_layer_norm: eqx.nn.LayerNorm
    encoder_attn: MultiHeadAttention
    encoder_attn_layer_norm: eqx.nn.LayerNorm
    ff: FeedForward
    final_layer_norm: eqx.nn.LayerNorm
    dropout: float

    def __init__(self, config, *, key):
        super().__init__()
        keys = jax.random.split(key, 3)

        self.self_attn = MultiHeadAttention(
            config.d_model, config.decoder_attention_heads, True, key=keys[0]
        )

        self.self_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.encoder_attn = MultiHeadAttention(
            config.d_model, config.decoder_attention_heads, True, key=keys[1]
        )

        self.encoder_attn_layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.ff = FeedForward(
            config.d_model,
            config.decoder_ffn_dim,
            config.activation_function,
            config.activation_dropout,
            key=keys[2],
        )

        self.final_layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def __call__(
        self,
        x: Float[Array, "s d"],
        encoder_hidden_states: Float[Array, "s d"],
        self_attn_mask: Optional[Float[Array, "1 s s"]],
        cross_attn_mask: Optional[Float[Array, "1 s s"]],
        *,
        key,
    ) -> Tuple[Float[Array, "s d"], Float[Array, "h s s"], Float[Array, "h s s"]]:
        keys = jax.random.split(key, 3)

        residual = x
        x = jax.vmap(self.self_attn_layer_norm)(x)
        x, self_attn = self.self_attn(x, attention_mask=self_attn_mask)
        x = eqx.nn.Dropout(self.dropout)(x, key=keys[0])
        x = residual + x

        residual = x
        x = jax.vmap(self.encoder_attn_layer_norm)(x)

        x, cross_attn = self.encoder_attn(
            x, key_value_states=encoder_hidden_states, attention_mask=cross_attn_mask
        )
        x = eqx.nn.Dropout(self.dropout)(x, key=keys[1])
        x = residual + x

        residual = x
        x = jax.vmap(self.final_layer_norm)(x)
        x = self.ff(x, key=keys[2])
        x = residual + x

        return x, self_attn, cross_attn


class WhisperDecoder(eqx.Module):
    embed_tokens: eqx.nn.Embedding
    embed_positions: SinusoidalPositionalEmbedding
    layers: list[DecoderLayer]
    layer_norm: eqx.nn.LayerNorm
    dropout: float

    def __init__(self, config, *, key):
        super().__init__()
        keys = jax.random.split(key, 3)

        self.embed_tokens = eqx.nn.Embedding(
            config.vocab_size, config.d_model, key=keys[0]
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_target_positions, config.d_model
        )
        self.layers = [
            DecoderLayer(config, key=k)
            for k in jax.random.split(keys[1], config.decoder_layers)
        ]
        self.layer_norm = eqx.nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def __call__(
        self,
        input_ids: Int[Array, " s"],
        encoder_hidden_states: Float[Array, "s d"],
        attention_mask: Optional[Float[Array, "1 s s"]],
        *,
        key,
    ) -> Float[Array, "s d"]:
        keys = jax.random.split(key, 2)

        # Direct embedding without vmap
        x = jax.vmap(self.embed_tokens)(input_ids)  # [s] -> [s, d]
        positions = self.embed_positions.weight[: x.shape[0]]  # [s, d]
        x = x + positions
        x = eqx.nn.Dropout(self.dropout)(x, key=keys[0])

        # Process through decoder layers
        for layer in self.layers:
            x, _, _ = layer(x, encoder_hidden_states, attention_mask, None, key=keys[1])

        return jax.vmap(self.layer_norm)(x)


class WhisperModel(eqx.Module):
    encoder: WhisperEncoder
    decoder: WhisperDecoder

    def __init__(self, config, *, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = WhisperEncoder(config, key=enc_key)
        self.decoder = WhisperDecoder(config, key=dec_key)

    def __call__(
        self,
        input_features: Float[Array, "f t"],
        decoder_input_ids: Int[Array, " s"],
        key: PRNGKeyArray,
    ) -> Float[Array, "s d"]:
        enc_key, dec_key = jax.random.split(key)
        encoder_out = self.encoder(input_features, key=enc_key)
        return self.decoder(decoder_input_ids, encoder_out, None, key=dec_key)


class WhisperForConditionalGeneration(eqx.Module):
    model: WhisperModel
    proj_out: eqx.nn.Linear

    def __init__(self, config, *, key):
        model_key, proj_key = jax.random.split(key)
        self.model = WhisperModel(config, key=model_key)
        self.proj_out = eqx.nn.Linear(
            config.d_model, config.vocab_size, use_bias=False, key=proj_key
        )

    def __call__(
        self,
        input_features: Float[Array, "b f t"],
        decoder_input_ids: Int[Array, "b s"],
        key,
    ) -> Float[Array, "b s v"]:
        decoder_out = self.model(input_features, decoder_input_ids, key=key)
        return jax.vmap(self.proj_out)(decoder_out)


class DecoderLayerWithCache(eqx.Module):
    self_attn: MultiHeadAttention
    self_attn_layer_norm: eqx.nn.LayerNorm
    encoder_attn: MultiHeadAttention
    encoder_attn_layer_norm: eqx.nn.LayerNorm
    ff: FeedForward
    final_layer_norm: eqx.nn.LayerNorm
    dropout: float

    def __call__(
        self,
        x: Float[Array, "b s d"],
        encoder_hidden_states: Optional[Float[Array, "b s d"]],
        self_attn_mask: Optional[Float[Array, "b 1 s s"]],
        cross_attn_mask: Optional[Float[Array, "b 1 s s"]],
        past_key_value: Optional[Tuple[Array, Array]],
        *,
        key: PRNGKeyArray,
    ) -> Tuple[Array, Tuple[Array, Array]]:
        keys = jax.random.split(key, 3)
        residual = x

        # Self attention with past
        x = self.self_attn_layer_norm(x)
        if past_key_value is None:
            attn_out, self_attn = self.self_attn(x, attention_mask=self_attn_mask)
        else:
            # Key-value concatenation for incremental decoding
            attn_out, self_attn = self.self_attn(
                x,
                key_value_states=jnp.concatenate([past_key_value[0], x], axis=1),
                attention_mask=self_attn_mask,
            )
        x = residual + eqx.nn.Dropout(self.dropout)(attn_out, key=keys[0])

        # Cross attention
        residual = x
        x = self.encoder_attn_layer_norm(x)
        cross_attn_out, cross_attn = self.encoder_attn(
            x, key_value_states=encoder_hidden_states, attention_mask=cross_attn_mask
        )
        x = residual + eqx.nn.Dropout(self.dropout)(cross_attn_out, key=keys[1])

        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.ff(x, key=keys[2])

        return x, (attn_out, cross_attn_out)
