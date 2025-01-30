# Whisper Equinox

Porting HF's Whisper for Equinox.

## Useful commands

- Prefix running any python command with `uv` via:

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
