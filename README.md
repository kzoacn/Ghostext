# Ghostext

[中文说明 / Chinese README](README.zh-CN.md)

Ghostext is a text steganography demo. It encrypts your message first, then hides the ciphertext inside language-model token choices so the same model, prompt, passphrase, and seed can recover it later.

The S&P 2027 research revision uses `guarded-prefix-v2`: independent random
suffix bits let the decoder recover a packet segment before its integer
interval collapses to a point. This addresses a measurable terminal sampling
bias in the historical codec. Both endpoints must use the same revision and
`--guard-bits` setting (default 128). `--guard-bits 0` selects the historical
codec for controlled comparisons. This change does not establish resistance
to text detectors, length analysis, or observation of retry behavior.
 
![Ghostext demo](imgs/demo.png)


## Quick start

Install from PyPI with local-LLM support:

```bash
pip install ghostext[llm]
```

Run a quick encode/decode round-trip check:

```bash
ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
| ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

See all command options:

```bash
ghostext --help
ghostext encode --help
ghostext decode --help
ghostext benchmark --help
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
ghostext encode \
  --prompt '请写一段自然、连贯、简短的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase 'river-pass' \
  --message '今晚七点在河边老地方见。'
```

`decode` reads the stego text from stdin by default, so a shell pipe stays short too:

```bash
ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
| ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

If you want to keep the generated text:

```bash
ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
  > stego.txt

ghostext decode \
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
- `--guard-bits`

The current default preset is:

- backend: `llama-cpp`
- seed: `7`
- ctx size: `4096`
- batch size: `128`
- top-p: `0.995`
- max candidates: `64`
- min entropy bits: `0.0`
- total frequency: `4096`
- guard bits: `128`

These are the values used when you do not override them.

## Common overrides

Use a model you already have:

```bash
ghostext encode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
| ghostext decode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

Change the default download directory:

```bash
GHOSTEXT_MODEL_DIR=/data/ghostext-models \
ghostext encode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
| GHOSTEXT_MODEL_DIR=/data/ghostext-models \
  ghostext decode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

Show structured metadata:

```bash
ghostext encode \
  --json \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!'
```

Disable logs when you want fully quiet output:

```bash
ghostext encode \
  --quiet \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!'
```

## Simple benchmark

Run a quick benchmark to measure:

- `encode latency`
- `decode latency`
- `encode bits/token`
- `ppl`

Use the local LLM backend (`llama-cpp`) with the same prompt/passphrase/message shown in the quick start:

```bash
ghostext benchmark \
  --backend llama-cpp \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
  --runs 3 \
  --json
```

## Reproducible 10x10 evaluation

If you want a larger artifact-style run instead of the built-in smoke benchmark,
the repo now includes a fixed 10 prompt x 10 message evaluation script.

Run it from a checkout of this repository:

```bash
python3 scripts/run_prompt_message_grid.py
```

Or use the small wrapper:

```bash
bash scripts/reproduce_prompt_message_grid.sh
```

For a quick sanity check before the full 100-trial run:

```bash
python3 scripts/run_prompt_message_grid.py --max-trials 1 --out-dir /tmp/ghostext-grid-smoke
```

By default this writes results under `results/prompt-message-grid/` and produces:

- `prompt_message_grid_dataset.json`
- `prompt_message_grid_runs.jsonl`
- `prompt_message_grid_summary.json`
- `prompt_message_grid_summary.md`

This evaluation uses the same local `llama-cpp` backend and default runtime
settings as the main CLI path, but it measures a broader fixed grid of public
prompts and messages instead of one README demo case.

## More detail

Use [spec.md](spec.md) if you want the protocol-level design and implementation notes. The README is intentionally optimized for getting from install to a working round-trip as quickly as possible.
