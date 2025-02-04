import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm
from transformers import GenerationConfig

from src.modelling import EquinoxWhisperModel
from src.utils import causal_mask

jax.config.update("jax_compilation_cache_dir", "/tmp/whisper_eqx_cache")
KEY = jax.random.PRNGKey(0)

# ruff: noqa: F722

class EquinoxWhisperGenerator:
    def __init__(self, model: EquinoxWhisperModel, generation_config: GenerationConfig):
        self.model = model
        self.config = generation_config
        self.generation_config = generation_config

    def _prepare_encoder_outputs(self, input_features: Array, mask: Array | None = None):
        return self.model.encoder(input_features, attn_mask=mask, key=jax.random.PRNGKey(0))

    def _process_logits(self, logits: Array, cursor: int):
        logits = logits[:, cursor - 1]  # Only need last token logits

        if self.generation_config.suppress_tokens:
            logits = logits.at[:, self.generation_config.suppress_tokens].set(-jnp.inf)

        return logits

    @eqx.filter_jit
    def _sample_next_token(
        self,
        decoder_input: Array,
        encoder_outputs: Array,
        token_idx: int,
        key: PRNGKeyArray,
    ) -> Array:
        _mask = jnp.where(decoder_input == 0, 0, 1)
        _causal_mask = causal_mask(decoder_input.shape[0], padding_mask=_mask).squeeze()

        logits = self.model.decoder(
            decoder_input, encoder_outputs, attention_mask=_causal_mask, key=key
        )

        logits = self.model.proj_out(logits)
        logits = self._process_logits(logits[None, ...], token_idx)

        if self.generation_config.do_sample:
            return jax.random.categorical(key, logits, axis=-1)
        else:
            return jnp.argmax(logits, axis=-1)

    @eqx.filter_jit
    def slow_generate(
        self, input_features: Float[Array, "batch mel time"]
    ) -> Int[Array, "batch seq"]:
        '''
        Legacy generate function
        '''
        maxlen = 32
        
        encoder_outputs = self._prepare_encoder_outputs(input_features)
        decoder_input = jnp.zeros(maxlen, dtype=jnp.int32) + self.config.pad_token_id
        decoder_input = decoder_input.at[0].set(self.config.decoder_start_token_id)

        for tok_idx in tqdm(range(1, maxlen)):
            next_token = self._sample_next_token(
                decoder_input,
                encoder_outputs,
                token_idx=tok_idx,
                key=KEY,
            )

            decoder_input = jax.lax.dynamic_update_slice(
                decoder_input,
                next_token,
                (tok_idx,)
            )

        return decoder_input

    @eqx.filter_jit
    def generate(self, input_features: Array):
        maxlen = 32
        
        encoder_outputs = self._prepare_encoder_outputs(input_features)
        decoder_input = jnp.zeros(maxlen, dtype=jnp.int32) + self.config.pad_token_id
        decoder_input = decoder_input.at[0].set(self.config.decoder_start_token_id)

        def body_fn(carry: Array, tok_idx: int):
            decoder_input = carry
            
            next_token = self._sample_next_token(
                decoder_input,
                encoder_outputs,
                token_idx=tok_idx,
                key=KEY
            )
            
            return jax.lax.dynamic_update_slice(decoder_input, next_token, (tok_idx,)), None

        final_decoder_input, _ = jax.lax.scan(
            body_fn,
            decoder_input,
            jnp.arange(1, maxlen), # type: ignore
            unroll=True
        )
        
        return final_decoder_input

