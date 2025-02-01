import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, PRNGKeyArray
from transformers import WhisperForConditionalGeneration

from src.main import EquinoxWhisperModel as EqxModel
from src.utils import causal_mask
from src.verify import convert_weights

KEY: PRNGKeyArray = jax.random.PRNGKey(0)
NUM_LAYERS: int = 4  # Adjust based on the model's actual number of decoder layers
TOLERANCE: float = 1e-2


def diff(
    eq_out: Array,
    pt_out: torch.Tensor | tuple[torch.Tensor],
    metadata: str | None = None,
):
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    max_diff = np.abs(eq_out - pt_out.detach().numpy()).max().item()
    assert max_diff <= TOLERANCE, f"Max difference: {max_diff} | {metadata}"


@pytest.fixture(scope="module")
def decoder_models() -> tuple[torch.nn.Module, eqx.Module]:
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en", attn_implementation="eager"
    )
    hf_model.eval()

    eqx_full_model = EqxModel(hf_model.config, key=KEY)
    eqx_full_model = convert_weights(hf_model, eqx_full_model)

    return hf_model.model.decoder, eqx_full_model.decoder


def test_decoder_embeddings(decoder_models):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long)

    # Token embeddings
    hf_tok = hf_model.embed_tokens(dummy_input)
    eqx_tok = jax.vmap(eqx_model.embed_tokens)(
        jnp.array(dummy_input.squeeze().numpy())
    )

    diff(eqx_tok, hf_tok, "token embeddings")

    # Positional embeddings
    positions = torch.arange(dummy_input.shape[1]).unsqueeze(0)
    hf_pos = hf_model.embed_positions(positions)
    eqx_pos = eqx_model.embed_positions.weight[: dummy_input.shape[1]]

    diff(eqx_pos, hf_pos.squeeze(0), "positional embeddings")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_q_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].self_attn.q_proj
    eqx_proj = eqx_model.layers[layer_idx].self_attn.q_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} self_attn q_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_k_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].self_attn.k_proj
    eqx_proj = eqx_model.layers[layer_idx].self_attn.k_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} self_attn k_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_v_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].self_attn.v_proj
    eqx_proj = eqx_model.layers[layer_idx].self_attn.v_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} self_attn v_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_out_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].self_attn.out_proj
    eqx_proj = eqx_model.layers[layer_idx].self_attn.out_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} self_attn out_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_q_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].encoder_attn.q_proj
    eqx_proj = eqx_model.layers[layer_idx].encoder_attn.q_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} cross_attn q_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_k_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].encoder_attn.k_proj
    eqx_proj = eqx_model.layers[layer_idx].encoder_attn.k_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} cross_attn k_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_v_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].encoder_attn.v_proj
    eqx_proj = eqx_model.layers[layer_idx].encoder_attn.v_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} cross_attn v_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_out_projection(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_proj = hf_model.layers[layer_idx].encoder_attn.out_proj
    eqx_proj = eqx_model.layers[layer_idx].encoder_attn.out_proj
    diff(
        eqx_proj(jnp.array(dummy_input.numpy())),
        hf_proj(dummy_input),
        f"layer {layer_idx} cross_attn out_proj",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_layer_norm(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_ln = hf_model.layers[layer_idx].self_attn_layer_norm
    eqx_ln = eqx_model.layers[layer_idx].self_attn_layer_norm
    diff(eqx_ln.weight, hf_ln.weight, f"layer {layer_idx} self_attn ln weight")
    diff(eqx_ln.bias, hf_ln.bias, f"layer {layer_idx} self_attn ln bias")
    diff(
        jax.vmap(eqx_ln)(jnp.array(dummy_input.numpy())),
        hf_ln(dummy_input),
        f"layer {layer_idx} self_attn ln output",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_layer_norm(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_ln = hf_model.layers[layer_idx].encoder_attn_layer_norm
    eqx_ln = eqx_model.layers[layer_idx].encoder_attn_layer_norm
    diff(eqx_ln.weight, hf_ln.weight, f"layer {layer_idx} cross_attn ln weight")
    diff(eqx_ln.bias, hf_ln.bias, f"layer {layer_idx} cross_attn ln bias")
    diff(
        jax.vmap(eqx_ln)(jnp.array(dummy_input.numpy())),
        hf_ln(dummy_input),
        f"layer {layer_idx} cross_attn ln output",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_ffn_fc1(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 384)
    hf_fc1 = hf_model.layers[layer_idx].fc1
    eqx_fc1 = eqx_model.layers[layer_idx].fc1
    diff(
        eqx_fc1(jnp.array(dummy_input.numpy())),
        hf_fc1(dummy_input),
        f"layer {layer_idx} fc1",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_ffn_fc2(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models
    dummy_input = torch.randn(5, 1536)
    hf_fc2 = hf_model.layers[layer_idx].fc2
    eqx_fc2 = eqx_model.layers[layer_idx].fc2

    diff(
        eqx_fc2(jnp.array(dummy_input.numpy())),
        hf_fc2(dummy_input),
        f"layer {layer_idx} fc2",
    )


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_self_attn_full(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models

    np.random.seed(0)
    dummy_input = torch.from_numpy(
        np.random.randn(1, 6, 384).astype(np.float32)
    )  # [num_mel_bins, time]

    hf_attn = hf_model.layers[layer_idx].self_attn
    eqx_attn = eqx_model.layers[layer_idx].self_attn

    hf_out, _, _ = hf_attn(dummy_input)
    eqx_out, _ = jax.vmap(eqx_attn)(jnp.array(dummy_input.numpy()))

    diff(eqx_out, hf_out, f"layer {layer_idx} self_attn output")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_cross_attn_full(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models

    dummy_input = torch.randn(1, 5, 384)
    dummy_encoder = torch.randn(1, 10, 384)

    hf_attn = hf_model.layers[layer_idx].encoder_attn
    eqx_attn = eqx_model.layers[layer_idx].encoder_attn

    hf_out, _, _ = hf_attn(dummy_input, key_value_states=dummy_encoder)
    eqx_out, _ = eqx_attn(
        jnp.array(dummy_input.numpy().squeeze()),
        key_value_states=jnp.array(dummy_encoder.numpy().squeeze()),
    )

    diff(eqx_out, hf_out, f"layer {layer_idx} cross_attn output")


@pytest.mark.parametrize("layer_idx", list(range(NUM_LAYERS)))
def test_decoder_layer_full(decoder_models, layer_idx: int):
    hf_model, eqx_model = decoder_models

    dummy_input = torch.randn(1, 5, 384)
    dummy_encoder = torch.randn(1, 10, 384)

    hf_layer = hf_model.layers[layer_idx]
    eqx_layer = eqx_model.layers[layer_idx]

    hf_out, _ = hf_layer(dummy_input, encoder_hidden_states=dummy_encoder)
    eqx_out, _, _ = eqx_layer(
        jnp.array(dummy_input.squeeze().numpy()),
        jnp.array(dummy_encoder.squeeze().numpy()),
        None,
        None,
        key=KEY,
    )

    diff(eqx_out, hf_out, f"layer {layer_idx} full forward")

@pytest.mark.parametrize("layer_pair", [(0, 1), (1, 2), (2, 3)])  # Adjust indices based on total layers
def test_consecutive_decoder_layers(decoder_models, layer_pair: tuple[int, int]):
    hf_model, eqx_model = decoder_models
    layer_idx1, layer_idx2 = layer_pair

    # Generate dummy inputs
    dummy_input = torch.randn(1, 5, 384)
    dummy_encoder = torch.randn(1, 10, 384)

    # Hugging Face forward pass
    hf_layer1 = hf_model.layers[layer_idx1]
    hf_layer2 = hf_model.layers[layer_idx2]
    hf_out1, _ = hf_layer1(dummy_input, encoder_hidden_states=dummy_encoder)
    hf_out2, _ = hf_layer2(hf_out1, encoder_hidden_states=dummy_encoder)

    # Equinox forward pass
    eqx_layer1 = eqx_model.layers[layer_idx1]
    eqx_layer2 = eqx_model.layers[layer_idx2]

    # Split keys for dropout in both layers
    key1, key2 = jax.random.split(KEY)

    # First layer
    eqx_out1, _, _ = eqx_layer1(
        jnp.array(dummy_input.squeeze().numpy()),
        jnp.array(dummy_encoder.squeeze().numpy()),
        None,  # self_attn_mask
        None,  # cross_attn_mask
        key=key1
    )

    # Second layer
    eqx_out2, _, _ = eqx_layer2(
        eqx_out1,  # Use output of first layer as input
        jnp.array(dummy_encoder.squeeze().numpy()),
        None,
        None,
        key=key2
    )

    # Compare final outputs
    diff(eqx_out2, hf_out2, f"consecutive layers {layer_idx1}-{layer_idx2} output")


def test_full_decoder(decoder_models):
    hf_model, eqx_model = decoder_models
    input_ids = torch.tensor([[50258, 50359]], dtype=torch.long)
    
    np.random.seed(0)
    dummy_encoder = np.random.randn(1, 10, 384)
    dummy_encoder = torch.Tensor(dummy_encoder)

    with torch.no_grad():
        hf_out = hf_model(
            input_ids, encoder_hidden_states=dummy_encoder
        ).last_hidden_state

    seq_len = input_ids.shape[-1]
    eqx_out = jax.vmap(eqx_model)(
        jnp.array(input_ids.numpy()),
        jnp.array(dummy_encoder.numpy()),
        causal_mask(seq_len),
        key=jax.random.split(KEY, 1),
    )
    diff(eqx_out, hf_out, "full decoder output")
