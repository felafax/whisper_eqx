# Whisper Equinox

Porting 🤗's `Whisper` implementation for Equinox.

## Outline

For this project, I wanted to take up `uv` so its built around it. Right now, all this has just been tested on `CPU` - so for `TPU` one may need to `uv add` the `TPU` versions of `JAX`.

As for the project structure (this could do with some cleanup):

- `run_e2e.py` actually runs the equinox and HF model end-to-end (i.e consumes audio and produces text) and verifies the outputs match up.

- `verify.py` is primilary to test whether the equinox port is correct vs. the HF model. It compares the `last_hidden_state` as that's more convenient.

This should be your first port-of-call whenever debugging any differences in the implementation. As a bonus, you also a get some statistics + a histogram of the deviations.

- `modelling.py` is the actual equinox port, the analogue of `modelling_whisper.py` used in the HF implementation and [WhisperJAX](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/modeling_flax_whisper.py).

Tests are pretty barebones, and exist for the `encoder` and `decoder` seperately.

A lot of optimizations (like KV caching) haven't been written yet. So this is more of a first-past port which adds JAX-friendly static-ness and relies on `XLA` for performance.

There's a lot of room for further speedups IMO.

## Useful commands

- Prefix running any (python) command with `uv` via:

```bash
uv run --env-file=.env <command>
```

- **Run tests:** Ensure you're at the project's root and run either:

```bash
uv run --env-file=.env pytest -s ./tests/test_encoder.py
uv run --env-file=.env pytest -s ./tests/test_decoder.py
```

For example, to verify the model outputs match up (e2e verification):

```bash
uv run --env-file=.env python3 ./src/verify.py
```

If `--env-file` feels cumbersome, you could just `export` its contents as well. I kept it incase we add some more envvars later (tokens and whatnot).
