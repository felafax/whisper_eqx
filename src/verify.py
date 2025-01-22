import jax
import jax.numpy as jnp
import torch
import numpy as np
import equinox as eqx
from transformers import WhisperModel
from main import WhisperModel as EqxModel


def convert_weights(hf_model, eqx_model):
    """Convert Hugging Face weights to Equinox model using proper immutable updates."""
    def convert_param(hf_param):
        return jnp.array(hf_param.detach().numpy().astype(np.float32))

    # Convert encoder components
    eqx_model = eqx.tree_at(
        lambda m: m.encoder.conv1.weight,
        eqx_model,
        convert_param(hf_model.encoder.conv1.weight)
    )

    eqx_model = eqx.tree_at(
        lambda m: m.encoder.conv1.bias,
        eqx_model,
        convert_param(hf_model.encoder.conv1.bias.unsqueeze(-1))
    )
    eqx_model = eqx.tree_at(
        lambda m: m.encoder.conv2.weight,
        eqx_model,
        convert_param(hf_model.encoder.conv2.weight)
    )
    eqx_model = eqx.tree_at(
        lambda m: m.encoder.conv2.bias,
        eqx_model,
        convert_param(hf_model.encoder.conv2.bias.unsqueeze(-1))
    )
    eqx_model = eqx.tree_at(
        lambda m: m.encoder.embed_positions.weight,
        eqx_model,
        convert_param(hf_model.encoder.embed_positions.weight)
    )

    # Convert encoder layers
    for layer_idx in range(len(hf_model.encoder.layers)):
        hf_layer = hf_model.encoder.layers[layer_idx]

        # Self attention
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.q_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.q_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.q_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.q_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.k_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.k_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.v_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.v_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.v_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.v_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.out_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.out_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn.out_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.out_proj.bias)
        )

        # Layer norms
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn_layer_norm.weight,
            eqx_model,
            convert_param(hf_layer.self_attn_layer_norm.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].self_attn_layer_norm.bias,
            eqx_model,
            convert_param(hf_layer.self_attn_layer_norm.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].final_layer_norm.weight,
            eqx_model,
            convert_param(hf_layer.final_layer_norm.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].final_layer_norm.bias,
            eqx_model,
            convert_param(hf_layer.final_layer_norm.bias)
        )

        # Feed forward
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].ff.fc1.weight,
            eqx_model,
            convert_param(hf_layer.fc1.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].ff.fc1.bias,
            eqx_model,
            convert_param(hf_layer.fc1.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].ff.fc2.weight,
            eqx_model,
            convert_param(hf_layer.fc2.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.encoder.layers[layer_idx].ff.fc2.bias,
            eqx_model,
            convert_param(hf_layer.fc2.bias)
        )

    # Convert decoder components
    eqx_model = eqx.tree_at(
        lambda m: m.decoder.embed_tokens.weight,
        eqx_model,
        convert_param(hf_model.decoder.embed_tokens.weight)
    )
    eqx_model = eqx.tree_at(
        lambda m: m.decoder.embed_positions.weight,
        eqx_model,
        convert_param(hf_model.decoder.embed_positions.weight)
    )

    # Convert decoder layers
    for layer_idx in range(len(hf_model.decoder.layers)):
        hf_layer = hf_model.decoder.layers[layer_idx]

        # Self attention
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.q_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.q_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.q_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.q_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.k_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.k_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.v_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.v_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.v_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.v_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.out_proj.weight,
            eqx_model,
            convert_param(hf_layer.self_attn.out_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].self_attn.out_proj.bias,
            eqx_model,
            convert_param(hf_layer.self_attn.out_proj.bias)
        )

        # Cross attention
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.q_proj.weight,
            eqx_model,
            convert_param(hf_layer.encoder_attn.q_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.q_proj.bias,
            eqx_model,
            convert_param(hf_layer.encoder_attn.q_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.k_proj.weight,
            eqx_model,
            convert_param(hf_layer.encoder_attn.k_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.v_proj.weight,
            eqx_model,
            convert_param(hf_layer.encoder_attn.v_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.v_proj.bias,
            eqx_model,
            convert_param(hf_layer.encoder_attn.v_proj.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.out_proj.weight,
            eqx_model,
            convert_param(hf_layer.encoder_attn.out_proj.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].encoder_attn.out_proj.bias,
            eqx_model,
            convert_param(hf_layer.encoder_attn.out_proj.bias)
        )

        # Layer norms
        for norm_type in ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"]:
            eqx_model = eqx.tree_at(
                lambda m: getattr(m.decoder.layers[layer_idx], norm_type).weight,
                eqx_model,
                convert_param(getattr(hf_layer, norm_type).weight)
            )
            eqx_model = eqx.tree_at(
                lambda m: getattr(m.decoder.layers[layer_idx], norm_type).bias,
                eqx_model,
                convert_param(getattr(hf_layer, norm_type).bias)
            )

        # Feed forward
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].ff.fc1.weight,
            eqx_model,
            convert_param(hf_layer.fc1.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].ff.fc1.bias,
            eqx_model,
            convert_param(hf_layer.fc1.bias)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].ff.fc2.weight,
            eqx_model,
            convert_param(hf_layer.fc2.weight)
        )
        eqx_model = eqx.tree_at(
            lambda m: m.decoder.layers[layer_idx].ff.fc2.bias,
            eqx_model,
            convert_param(hf_layer.fc2.bias)
        )

    return eqx_model


def test_equivalence():
    hf_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    hf_model.eval()
    
    # Create Equinox model with random initialization
    eqx_model = EqxModel(hf_model.config, key=jax.random.PRNGKey(0))
    
    # Perform weight conversion
    eqx_model = convert_weights(hf_model, eqx_model)

    # Create test input
    batch_size = 1
    seq_len = 1024
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    input_features = np.random.randn(batch_size, hf_model.config.num_mel_bins, seq_len)
    decoder_input_ids = np.array([[50258, 50359]], dtype=np.int32)
    decoder_input_ids = np.broadcast_to(decoder_input_ids, (batch_size, 2))

    with torch.no_grad():
        hf_outputs = hf_model(
            torch.from_numpy(input_features).float(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids)
        )

    hf_last_hidden = hf_outputs.last_hidden_state.numpy()

    # Run Equinox model
    eqx_input = jnp.array(input_features.astype(np.float32))
    eqx_decoder_input = jnp.array(decoder_input_ids)

    eqx_out = eqx.filter_vmap(eqx_model)(eqx_input, eqx_decoder_input, keys)

    # Compare outputs
    print("\nModel Output Comparison:")
    print(f"HF output shape: {hf_last_hidden.shape}")
    print(f"Equinox output shape: {eqx_out.shape}")
    print(f"Max absolute difference: {jnp.max(jnp.abs(eqx_out - hf_last_hidden)):.2e}")
    print(f"Mean absolute difference: {jnp.mean(jnp.abs(eqx_out - hf_last_hidden)):.2e}")

    # Verify numerical equivalence
    assert jnp.allclose(eqx_out, hf_last_hidden, atol=1e-4), \
        "Model outputs differ significantly! Check weight conversion."


if __name__ == "__main__":
    test_equivalence()
