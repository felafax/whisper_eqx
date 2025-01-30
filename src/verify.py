import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from transformers import WhisperModel

from src.main import WhisperModel as EqxModel
from src.utils import plot_deviation_histogram


def parse_path(path_str):
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


def get_hf_param(hf_model, path_str):
    parts = parse_path(path_str)
    current = hf_model
    for part in parts:
        if isinstance(part, int):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def create_where_func(path_str):
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
        ]
    ):
        param_jax = param_jax.T

    # Unsqueeze encoder conv biases (Equinox expects shape [out_channels, 1])
    if ("conv1.bias" in path or "conv2.bias" in path) and "encoder" in path:
        param_jax = jnp.expand_dims(param_jax, axis=-1)

    return param_jax


def update_param(eqx_model, hf_model, path_str):
    where = create_where_func(path_str)
    hf_param = get_hf_param(hf_model, path_str)
    new_param = process_param(hf_param, path_str)
    return eqx.tree_at(where, eqx_model, new_param)


def convert_weights(hf_model: torch.nn.Module, eqx_model: eqx.Module) -> eqx.Module:
    """Convert Hugging Face weights to Equinox model using path-based transformations."""

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
    hf_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    hf_model.eval()

    # Create Equinox model with random initialization
    assert hf_model.config.dropout == 0.0, (
        f"Ensure non-zero dropout. Got: {hf_model.config.dropout}"
    )

    eqx_model = EqxModel(hf_model.config, key=jax.random.PRNGKey(0))

    # Perform weight conversion
    eqx_model = convert_weights(hf_model, eqx_model)

    # Create test input
    batch_size = 1
    seq_len = 3000
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    input_features = np.random.randn(batch_size, hf_model.config.num_mel_bins, seq_len)
    decoder_input_ids = np.array([[50258, 50359]], dtype=np.int32)
    decoder_input_ids = np.broadcast_to(decoder_input_ids, (batch_size, 2))

    with torch.no_grad():
        hf_outputs = hf_model(
            torch.from_numpy(input_features).float(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids),
        )

    hf_last_hidden = hf_outputs.last_hidden_state.numpy()

    # Run Equinox model
    eqx_input = jnp.array(input_features.astype(np.float32))
    eqx_decoder_input = jnp.array(decoder_input_ids)

    eqx_out = eqx.filter_vmap(eqx_model)(eqx_input, eqx_decoder_input, keys)  # type: ignore

    # Compare outputs
    print("\nModel Output Comparison:")
    print(f"HF output shape: {hf_last_hidden.shape}")
    print(f"Equinox output shape: {eqx_out.shape}")
    print(f"Max absolute difference: {jnp.max(jnp.abs(eqx_out - hf_last_hidden)):.2e}")
    print(
        f"Mean absolute difference: {jnp.mean(jnp.abs(eqx_out - hf_last_hidden)):.2e}"
    )

    # Verify numerical equivalence
    assert jnp.allclose(eqx_out, hf_last_hidden, atol=5e-1), (
        "Model outputs differ significantly! Check weight conversion."
    )

def ascii_hist(x, bins=10):
    n, xedges = np.histogram(x, bins=bins)
    max_n = n.max()
    bar_width = 50
    if max_n > 0:
        normed_n = n / max_n
    else:
        normed_n = n  # Avoid division by zero if max_n is zero

    for i in range(len(n)):
        bar = '#' * int(normed_n[i] * bar_width)
        bin_center = (xedges[i] + xedges[i + 1]) / 2
        bin_str = '{0: <8.4g}'.format(bin_center).ljust(10)
        count_str = f"({n[i]})"
        print(f'{bin_str}| {bar} {count_str}')



def test_equivalence_better():
    hf_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    hf_model.eval()

    # Create Equinox model with random initialization
    assert hf_model.config.dropout == 0.0, (
        f"Ensure non-zero dropout. Got: {hf_model.config.dropout}"
    )

    eqx_model = EqxModel(hf_model.config, key=jax.random.PRNGKey(0))

    # Perform weight conversion
    eqx_model = convert_weights(hf_model, eqx_model)

    # Create test input
    batch_size = 1
    seq_len = 3000
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    input_features = np.random.randn(batch_size, hf_model.config.num_mel_bins, seq_len)
    decoder_input_ids = np.array([[50258, 50359]], dtype=np.int32)
    decoder_input_ids = np.broadcast_to(decoder_input_ids, (batch_size, 2))

    with torch.no_grad():
        hf_outputs = hf_model(
            torch.from_numpy(input_features).float(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids),
        )

    hf_last_hidden = hf_outputs.last_hidden_state.numpy()

    # Run Equinox model
    eqx_input = jnp.array(input_features.astype(np.float32))
    eqx_decoder_input = jnp.array(decoder_input_ids)

    eqx_out = eqx.filter_vmap(eqx_model)(eqx_input, eqx_decoder_input, keys)  # type: ignore

    plot_deviation_histogram(hf_last_hidden, eqx_out)

    # Verify numerical equivalence
    assert jnp.allclose(eqx_out, hf_last_hidden, atol=5e-1), (
        "Model outputs differ significantly! Check weight conversion."
    )

if __name__ == '__main__':
    test_equivalence_better()
