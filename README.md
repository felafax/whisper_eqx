# Whisper Equinox

Porting HF's Whisper for Equinox.

## Useful commands


- **Run tests:** Ensure you're at the project's root and do:

```bash
uv run --env-file=.env pytest -s ./tests/test_encoder.py
```

- Prefix running any python command with `uv` via:

```bash
uv run --env-file=.env <command>
```

For example, to verify the model outputs match up:

```bash
uv run --env-file=.env python3 ./src/verify.py
```
