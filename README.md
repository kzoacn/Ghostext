# Ghostext-codex

[中文说明 / Chinese README](README.zh-CN.md)

Ghostext is a deterministic text steganography demo. It encrypts your message first, then hides the ciphertext inside language-model token choices so the same model, prompt, passphrase, and seed can recover it later.

The default user path is now the real local `llama.cpp` backend. The toy backend still exists, but it is meant for tests and protocol debugging.

## Quick start

Install the package with local-LLM support:

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[llm]'
```

Run a quick encode/decode round-trip check:

```bash
.venv/bin/ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| .venv/bin/ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

See all command options:

```bash
.venv/bin/ghostext --help
.venv/bin/ghostext encode --help
.venv/bin/ghostext decode --help
```

On the first run, Ghostext will:

- look for a local model from `--model-path`
- otherwise check `GHOSTEXT_MODEL_PATH` or `GHOSTEXT_LLAMA_MODEL_PATH`
- otherwise download the default GGUF model to `~/.cache/ghostext/models/qwen35-2b-q4ks/`

The default model is:

- model id: `Qwen/Qwen3.5-2B`
- repo: `bartowski/Qwen_Qwen3.5-2B-GGUF`
- file: `Qwen_Qwen3.5-2B-Q4_K_S.gguf`

## Everyday usage

Most of the time, `encode` only needs these arguments:

```bash
.venv/bin/ghostext encode \
  --prompt '请写一段自然、连贯、简短的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase 'river-pass' \
  --message '今晚七点在河边老地方见。'
```

`decode` reads the stego text from stdin by default, so a shell pipe stays short too:

```bash
.venv/bin/ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| .venv/bin/ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

If you want to keep the generated text:

```bash
.venv/bin/ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
  > stego.txt

.venv/bin/ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --text-file stego.txt
```

## Defaults You No Longer Need To Tune

The CLI now comes with a real-model preset so users usually do not need to care about:

- `--seed` unless you want a custom shared seed
- `--ctx-size`
- `--batch-size`
- `--top-p`
- `--max-candidates`
- `--min-entropy-bits`
- `--totfreq`

The current default preset is:

- backend: `llama-cpp`
- seed: `7`
- ctx size: `4096`
- batch size: `128`
- top-p: `0.995`
- max candidates: `64`
- min entropy bits: `0.0`
- total frequency: `4096`

These are the values used when you do not override them.

## Common overrides

Use a model you already have:

```bash
.venv/bin/ghostext encode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| .venv/bin/ghostext decode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

Change the default download directory:

```bash
GHOSTEXT_MODEL_DIR=/data/ghostext-models \
.venv/bin/ghostext encode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| GHOSTEXT_MODEL_DIR=/data/ghostext-models \
  .venv/bin/ghostext decode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

Show structured metadata:

```bash
.venv/bin/ghostext encode \
  --json \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

Disable logs when you want fully quiet output:

```bash
.venv/bin/ghostext encode \
  --quiet \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

## Toy backend

`ToyCharBackend` is still available, but it is for tests and protocol experiments rather than normal use:

```bash
.venv/bin/ghostext encode \
  --backend toy \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase test-pass \
  --message 'toy backend roundtrip' \
| .venv/bin/ghostext decode \
  --backend toy \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase test-pass
```

## Tests

Run the current test suite with:

```bash
env PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The real-model smoke test stays opt-in because it needs a local GGUF model:

```bash
GHOSTEXT_LLAMA_MODEL_PATH=/abs/path/to/model.gguf \
env PYTHONPATH=src python3 -m unittest tests.test_llama_cpp_integration -v
```

## More detail

Use [spec.md](spec.md) if you want the protocol-level design and implementation notes. The README is intentionally optimized for getting from install to a working round-trip as quickly as possible.
