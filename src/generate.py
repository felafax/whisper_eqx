import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from tqdm import tqdm
from transformers import GenerationConfig

from src.main import EquinoxWhisperModel
from src.utils import causal_mask

# ruff: noqa: F722

class EquinoxWhisperGenerator:
    def __init__(self, model: EquinoxWhisperModel, generation_config: GenerationConfig):
        self.model = model
        self.config = generation_config
        self.generation_config = generation_config

    def _prepare_encoder_outputs(self, input_features: Array):
        return self.model.encoder(input_features, key=jax.random.PRNGKey(0))

    def _get_initial_cache(self, encoder_outputs: Array, batch_size: int):
        # For simplicity, we'll use full sequence processing initially
        # For production, implement proper KV caching
        return None

    def _process_logits(self, logits: Array):
        logits = logits[:, -1]  # Only need last token logits

        if self.generation_config.suppress_tokens:
            logits = logits.at[:, self.generation_config.suppress_tokens].set(-jnp.inf)

        return logits

    def _sample_next_token(self, logits: Array) -> Array:
        key = jax.random.PRNGKey(0)

        if self.generation_config.do_sample:
            return jax.random.categorical(key, logits, axis=-1)
        else:
            return jnp.argmax(logits, axis=-1)

    def generate(
        self, input_features: Float[Array, "batch mel time"]
    ) -> Int[Array, "batch seq"]:
        decoder_input = jnp.array([self.config.decoder_start_token_id], dtype=jnp.int32)
        encoder_outputs = self._prepare_encoder_outputs(input_features)

        key = jax.random.PRNGKey(0)  # Initialize a root key
        maxlen = 32

        for _ in tqdm(range(maxlen)):
            current_seq_len = decoder_input.shape[-1]
            _causal_mask = causal_mask(
                current_seq_len
            ).squeeze()  # Generate causal mask

            # Forward pass with updated causal mask
            logits = self.model.decoder(
                decoder_input, encoder_outputs, attention_mask=_causal_mask, key=key
            )

            logits = self.model.proj_out(logits)
            logits = self._process_logits(logits[None, ...])

            # Split key for sampling
            next_tokens = self._sample_next_token(logits)

            decoder_input = jnp.concatenate([decoder_input, next_tokens], axis=-1)

            if jnp.all(next_tokens == self.config.eos_token_id):
                break

        return decoder_input
