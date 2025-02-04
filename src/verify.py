from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from transformers import WhisperForConditionalGeneration

from src.main import EquinoxWhisperModel as EqxModel
from src.utils import create_where_func, get_hf_param, plot_deviation_histogram

T = TypeVar('T')

def process_param(hf_param: torch.Tensor, path: str) -> Array:
    param_np = hf_param.detach().numpy().astype(np.float32)
    param_jax = jnp.array(param_np)

    # Transpose linear layers (FFN and MHA projections)
    if any(
        key in path
        for key in [
            "fc1.weight",
            "fc2.weight",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
            "proj_out.weight",
        ]
    ):
        param_jax = param_jax.T

    # Unsqueeze encoder conv biases (Equinox expects shape [out_channels, 1])
    if ("conv1.bias" in path or "conv2.bias" in path) and "encoder" in path:
        param_jax = jnp.expand_dims(param_jax, axis=-1)

    return param_jax


def update_param(
    eqx_model: eqx.Module, hf_model: torch.nn.Module, path_str: str
) -> eqx.Module:
    '''
    Cleanly wrap ugly & repetitive `tree_at` calls.
    '''
    
    where = create_where_func(path_str)
    hf_param = get_hf_param(hf_model, path_str)
    new_param = process_param(hf_param, path_str)
    
    return eqx.tree_at(where, eqx_model, new_param)


def convert_weights(hf_for_gen: torch.nn.Module, eqx_model: T) -> T:
    """Convert Hugging Face weights to Equinox model using path-based transformations."""
    assert isinstance(hf_for_gen, WhisperForConditionalGeneration), 'Need correct HF class for output projection loading.'
    assert isinstance(eqx_model, EqxModel), 'Provide generator for correct porting.'
    
    hf_model = hf_for_gen.model

    # Encoder components
    eqx_model = update_param(eqx_model, hf_model, "encoder.conv1.weight")
    eqx_model = update_param(eqx_model, hf_model, "encoder.conv1.bias")
    eqx_model = update_param(eqx_model, hf_model, "encoder.conv2.weight")
    eqx_model = update_param(eqx_model, hf_model, "encoder.conv2.bias")
    eqx_model = update_param(eqx_model, hf_model, "encoder.embed_positions.weight")
    eqx_model = update_param(eqx_model, hf_model, "encoder.layer_norm.weight")
    eqx_model = update_param(eqx_model, hf_model, "encoder.layer_norm.bias")

    # Encoder layers
    for layer_idx in range(len(hf_model.encoder.layers)):
        base_path = f"encoder.layers[{layer_idx}]"

        # Self attention
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.q_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.q_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.k_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.v_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.v_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.out_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.out_proj.bias"
        )

        # Layer norms
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn_layer_norm.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn_layer_norm.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.final_layer_norm.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.final_layer_norm.bias"
        )

        # Feed forward
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc1.weight")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc1.bias")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc2.weight")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc2.bias")

    # Decoder components
    eqx_model = update_param(eqx_model, hf_model, "decoder.embed_tokens.weight")
    eqx_model = update_param(eqx_model, hf_model, "decoder.embed_positions.weight")

    # Decoder `layer_norm` at the end
    eqx_model = update_param(eqx_model, hf_model, "decoder.layer_norm.weight")
    eqx_model = update_param(eqx_model, hf_model, "decoder.layer_norm.bias")

    # Output projection
    eqx_model = update_param(eqx_model, hf_for_gen, 'proj_out.weight')

    # Decoder layers
    for layer_idx in range(len(hf_model.decoder.layers)):
        base_path = f"decoder.layers[{layer_idx}]"

        # Self attention
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.q_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.q_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.k_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.v_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.v_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.out_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.self_attn.out_proj.bias"
        )

        # Cross attention
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.q_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.q_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.k_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.v_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.v_proj.bias"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.out_proj.weight"
        )
        eqx_model = update_param(
            eqx_model, hf_model, f"{base_path}.encoder_attn.out_proj.bias"
        )

        # Layer norms
        for norm_type in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ]:
            eqx_model = update_param(
                eqx_model, hf_model, f"{base_path}.{norm_type}.weight"
            )
            eqx_model = update_param(
                eqx_model, hf_model, f"{base_path}.{norm_type}.bias"
            )

        # Feed forward
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc1.weight")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc1.bias")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc2.weight")
        eqx_model = update_param(eqx_model, hf_model, f"{base_path}.fc2.bias")

    return eqx_model


def test_equivalence():
    hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    hf_model.eval()

    eqx_model = EqxModel(hf_model.config, key=jax.random.PRNGKey(0))
    eqx_model = convert_weights(hf_model, eqx_model)

    # Create test input
    batch_size = 1
    seq_len = 3000
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    input_features = np.random.randn(batch_size, hf_model.config.num_mel_bins, seq_len)
    decoder_input_ids = np.array([[50258, 50359]], dtype=np.int32)
    decoder_input_ids = np.broadcast_to(decoder_input_ids, (batch_size, 2))

    with torch.no_grad():
        hf_outputs = hf_model.model(
            torch.from_numpy(input_features).float(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids),
        )

    hf_last_hidden = hf_outputs.last_hidden_state.numpy()

    # Run Equinox model
    eqx_input = jnp.array(input_features.astype(np.float32))
    eqx_decoder_input = jnp.array(decoder_input_ids)
    eqx_out, eqx_last_hidden = eqx.filter_vmap(eqx_model)(eqx_input, eqx_decoder_input, keys)  # type: ignore

    plot_deviation_histogram(hf_last_hidden, eqx_last_hidden)

    # Verify numerical equivalence
    assert jnp.allclose(eqx_last_hidden, hf_last_hidden, atol=5e-1), (
        "Model outputs differ significantly! Check weight conversion."
    )

if __name__ == '__main__':
    test_equivalence()
