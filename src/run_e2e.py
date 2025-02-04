import jax
import jax.numpy as jnp
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperProcessor

from src.generate import EquinoxWhisperGenerator
from src.modelling import EquinoxWhisperModel as EqxModel
from src.verify import convert_weights


def test_e2e():
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    audio_sample = ds[0]["audio"]

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    hf_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    gen_hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    hf_model.eval()
    key = jax.random.PRNGKey(0)
    
    input_features: torch.Tensor = processor(
        audio_sample["array"],
        sampling_rate=audio_sample["sampling_rate"],
        return_tensors="pt",
    ).input_features.float()
   
    eqx_model = EqxModel(hf_model.config, key=key)
    eqx_model = convert_weights(gen_hf_model, eqx_model)
   
    hf_generated = gen_hf_model.generate(input_features)
    hf_decoded = processor.batch_decode(hf_generated.tolist(), skip_special_tokens=True) 
    print(f'HF model output: {hf_decoded} | Generated: {hf_generated}')
   
    eqx_generator = EquinoxWhisperGenerator(eqx_model, hf_model.config)
    eqx_generated = eqx_generator.generate(jnp.array(input_features.squeeze().numpy()))
    eqx_decoded = processor.batch_decode([eqx_generated.tolist()], skip_special_tokens=True)
    print(f'Equinox model output: {eqx_decoded} | Generated: {eqx_generated}')

    assert eqx_decoded == hf_decoded, "Outputs don't match up!"
    
if __name__ == '__main__':
    test_e2e()
