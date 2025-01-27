import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, PRNGKeyArray
from transformers import WhisperModel

from src.main import WhisperModel as EqxModel
from src.verify import convert_weights

KEY: PRNGKeyArray = jax.random.PRNGKey(0)
NUM_LAYERS: int = 4
TOLERANCE: float = 1e-4


def diff(
    eq_out: Array,
    pt_out: torch.Tensor | tuple[torch.Tensor],
    metadata: str | None = None,
):
    """
    Ensure that both provided inputs are close to each other
    """
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]

    max_diff = np.abs(eq_out - pt_out.detach().numpy()).max().item()
    assert max_diff <= TOLERANCE, f"Max difference: {max_diff} | {metadata}"


@pytest.fixture(scope="module")
def models() -> tuple[torch.nn.Module, eqx.Module]:
    hf_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    hf_model.eval()

    # Create full Equinox model first
    eqx_full_model = EqxModel(hf_model.config, key=KEY)
    eqx_full_model = convert_weights(hf_model, eqx_full_model)

    # Extract encoder from converted model
    eqx_encoder = eqx_full_model.encoder  # type: ignore

    return hf_model, eqx_encoder


def test_conv1(models):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(80, 3000)

    hf_out = hf_model.encoder.conv1(dummy_input)
    eq_out = eqx_encoder.conv1(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out)


def test_conv2(models):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(384, 3000)

    hf_out = hf_model.encoder.conv2(dummy_input)
    eq_out = eqx_encoder.conv2(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out)


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_self_attn_k_projection(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_k_proj = hf_model.encoder.layers[layer_idx].self_attn.k_proj
    eq_k_proj = eqx_encoder.layers[layer_idx].self_attn.k_proj

    hf_out = hf_k_proj(dummy_input)
    eq_out = eq_k_proj(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out, f"layer {layer_idx} k_proj")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_self_attn_v_projection(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_v_proj = hf_model.encoder.layers[layer_idx].self_attn.v_proj
    eq_v_proj = eqx_encoder.layers[layer_idx].self_attn.v_proj

    hf_out = hf_v_proj(dummy_input)
    eq_out = eq_v_proj(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out, f"layer {layer_idx} v_proj")

@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_ffn_output(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_layer = hf_model.encoder.layers[layer_idx]
    eq_layer = eqx_encoder.layers[layer_idx]

    # HF forward
    with torch.no_grad():
        hf_out = hf_layer.fc2(hf_layer.fc1(dummy_input))
    
    # Equinox forward
    eq_out = eq_layer.ff.fc2(eq_layer.ff.fc1(jnp.array(dummy_input.numpy())))
    
    diff(eq_out, hf_out, f"layer {layer_idx} FFN output")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_self_attn_out_projection(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_out_proj = hf_model.encoder.layers[layer_idx].self_attn.out_proj
    eq_out_proj = eqx_encoder.layers[layer_idx].self_attn.out_proj

    hf_out = hf_out_proj(dummy_input)
    eq_out = eq_out_proj(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out, f"layer {layer_idx} out_proj")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_self_attn_full_forward(models, layer_idx: int):
    """
    Test the entire multi-head self-attention sub-layer forward pass,
    comparing Hugging Face vs. Equinox outputs.
    """
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(1, 6, 384)

    # HF MHA sub-layer
    hf_self_attn = hf_model.encoder.layers[layer_idx].self_attn
    # Equinox MHA sub-layer
    eq_self_attn = eqx_encoder.layers[layer_idx].self_attn

    hf_out, hf_attn_weights, _ = hf_self_attn(
        hidden_states=dummy_input, output_attentions=True
    )

    # For Equinox, we call eq_self_attn with jnp arrays:
    eq_out, eq_attn_weights = jax.vmap(eq_self_attn)(
        jnp.array(dummy_input.numpy()), attention_mask=None
    )

    diff(eq_out, hf_out, f"layer {layer_idx} MHA output")
    diff(eq_attn_weights, hf_attn_weights, f"layer {layer_idx} MHA attention weights")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_self_attn_projection(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_layer = hf_model.encoder.layers[layer_idx].self_attn.q_proj
    eq_layer = eqx_encoder.layers[layer_idx].self_attn.q_proj

    hf_out = hf_layer(dummy_input)
    eq_out = eq_layer(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out)


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_attention_layer_norm(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_ln = hf_model.encoder.layers[layer_idx].self_attn_layer_norm
    eq_ln = eqx_encoder.layers[layer_idx].self_attn_layer_norm

    diff(eq_ln.weight, hf_ln.weight, f"Weight of ln: {layer_idx}")
    diff(eq_ln.bias, hf_ln.bias, f"Bias of ln: {layer_idx}")

    hf_out = hf_ln(dummy_input)
    eq_out = jax.vmap(eq_ln)(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out)


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_ffn(models, layer_idx: int):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(10, 384)

    hf_ff = hf_model.encoder.layers[layer_idx].fc1
    eq_ff = eqx_encoder.layers[layer_idx].ff.fc1

    hf_out = hf_ff(dummy_input)
    eq_out = eq_ff(jnp.array(dummy_input.numpy()))

    diff(eq_out, hf_out)


def test_full_encoder(models):
    hf_model, eqx_encoder = models
    dummy_input = torch.randn(1, 80, 3000)  # [num_mel_bins, time]

    hf_out = hf_model.encoder(dummy_input).last_hidden_state
    eq_out = eqx_encoder(jnp.array(dummy_input.numpy().squeeze()), key=KEY)[None, ...]

    diff(eq_out, hf_out)
