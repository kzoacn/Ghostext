# HideText-codex

[中文说明 / Chinese README](README.zh-CN.md)

HideText is a deterministic text steganography demo that hides an encrypted payload inside next-token choices during language-model generation. Instead of editing a finished paragraph, the sender maps ciphertext bits into token selections under a shared model, tokenizer, prompt, seed, and codec configuration; the receiver then replays the same distributions to recover the payload.

The project is intentionally optimized for reproducibility and decodability before naturalness and capacity.

## What is implemented

- fixed packet framing with a runtime config fingerprint
- `scrypt + ChaCha20-Poly1305` encrypt-then-stego payload construction
- deterministic candidate selection and integer probability quantization
- an integer finite-interval header/body codec
- a bilingual deterministic toy backend for fast protocol tests
- a `llama.cpp` backend for local Qwen GGUF inference on CPU
- CLI commands for `encode`, `decode`, and `eval`
- fail-closed negative tests for prompt drift, seed drift, text mutation, and retry exhaustion
- low-entropy monitoring with automatic re-attempts
- natural tail generation after the secret-bearing prefix is complete

## Quick start

Create a local environment and install the base package:

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

Run the default test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Real-model setup

Install the optional local-LLM dependencies:

```bash
.venv/bin/python -m pip install -e '.[llm]'
```

Download the verified GGUF model:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    local_dir="models/qwen3-4b-q4km",
)
print(path)
PY
```

Run the real integration smoke test:

```bash
HIDETEXT_LLAMA_MODEL_PATH=/abs/path/to/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
HIDETEXT_LLAMA_THREADS=8 \
HIDETEXT_LLAMA_CTX=4096 \
HIDETEXT_LLAMA_BATCH=128 \
.venv/bin/python -m unittest tests.test_llama_cpp_integration -v
```

The repository was validated locally with:

- the default `unittest` suite over the toy backend
- a real CPU smoke test using `Qwen/Qwen3-4B-Instruct-2507` in GGUF `Q4_K_M` form

## Low-entropy retry behavior

Encoding can fail when the model collapses into a near-deterministic region and the candidate entropy effectively drops to zero. HideText now guards against this by default:

- it monitors a rolling window of `32` consecutive token steps
- if the average candidate entropy over that window falls below `0.1` bit, the current encoding attempt is abandoned
- a fresh attempt is started with a new random packet salt/nonce, which changes the encrypted body and therefore the token path
- after `3` failed attempts, HideText raises an explicit error suggesting a different cover prompt or a shorter secret

These knobs are sender-side runtime safety settings and do not enter the packet fingerprint:

- `--low-entropy-window-tokens`
- `--low-entropy-threshold-bits`
- `--max-encode-attempts`
- `--stall-patience-tokens`

Set `--low-entropy-window-tokens 0` to disable the low-entropy detector for controlled experiments with a fixed packet.

## Natural ending

HideText no longer has to stop exactly at the token where the packet becomes decodable. Once the header and body are fully embedded:

- the secret-bearing prefix is complete and decodable
- the sender may continue sampling a short random tail for a more natural-looking ending
- this tail is not part of the codec and is ignored by the decoder
- decoding stops as soon as the packet is fully recovered, even if more text follows

Use `--natural-tail-max-tokens` to cap how many post-packet tokens may be sampled. Set it to `0` if you want the old stop-immediately behavior.

## CLI usage

Toy backend round-trip:

```bash
.venv/bin/python -m hidetext.cli eval \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase pass-en \
  --message 'Meet me near the station at seven.' \
  --seed 29
```

Real Qwen CPU round-trip:

```bash
.venv/bin/python -m hidetext.cli eval \
  --backend llama-cpp \
  --model-path models/qwen3-4b-q4km/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --threads 8 \
  --ctx-size 4096 \
  --batch-size 128 \
  --top-p 0.995 \
  --max-candidates 64 \
  --min-entropy-bits 0.0 \
  --totfreq 4096 \
  --header-token-budget 1024 \
  --body-token-budget 4096 \
  --natural-tail-max-tokens 64 \
  --stall-patience-tokens 256 \
  --low-entropy-window-tokens 32 \
  --low-entropy-threshold-bits 0.1 \
  --max-encode-attempts 3 \
  --prompt 'Please write a short, natural paragraph about a quiet evening walk.' \
  --passphrase real-pass \
  --message 'Meet near the riverside at seven.' \
  --seed 7
```

If you want file-based inputs:

```bash
.venv/bin/python -m hidetext.cli eval \
  --prompt-file prompts/zh.txt \
  --passphrase-file secrets/passphrase.txt \
  --seed-file secrets/seed.txt \
  --message '今晚七点在老地方见。'
```

## Examples

### English example

- Backend: `toy`
- Prompt / cover prompt: `Write a calm and readable English paragraph about a quiet evening walk.`
- Secret message: `Meet near the riverside at seven.`
- Passphrase: `demo-en`
- Seed: `29`

Command:

```bash
.venv/bin/python -m hidetext.cli encode \
  --prompt 'Write a calm and readable English paragraph about a quiet evening walk.' \
  --passphrase demo-en \
  --message 'Meet near the riverside at seven.' \
  --seed 29
```

Stego text excerpt:

```text
The afternoon feels cllm, and the corner cafe keeps a warm light on while people talk softlv along the street. A quiet breeze moves past the window, the pages stay hflf open on tte table, and the room keeps a steady brightness.
```

### Chinese example

- Backend: `toy`
- Prompt / cover prompt: `请写一段自然、柔和、连贯的中文短文，描写傍晚散步时看到的街景。`
- Secret message: `今晚七点在河边老地方见。`
- Passphrase: `demo-zh`
- Seed: `17`

Command:

```bash
.venv/bin/python -m hidetext.cli encode \
  --prompt '请写一段自然、柔和、连贯的中文短文，描写傍晚散步时看到的街景。' \
  --passphrase demo-zh \
  --message '今晚七点在河边老地方见。' \
  --seed 17
```

Stego text excerpt:

```text
今天的风很轻，街角的咖啡店还亮着暖黄的静，路过的人慢慢聊着天，城天显得安静而柔和。午后的窗边有一轻淡淡香茶的，桌上书书翻到一半，雨声落在树叶上，房间里有种稳稳的明亮。
```

## Repository layout

```text
src/hidetext/
  candidate_policy.py
  cli.py
  codec.py
  config.py
  crypto.py
  decoder.py
  encoder.py
  errors.py
  llama_cpp_backend.py
  model_backend.py
  packet.py
  pipeline.py
  progress.py
  quantization.py
tests/
  test_cli.py
  test_codec_toy.py
  test_crypto.py
  test_failures.py
  test_llama_cpp_integration.py
  test_packet.py
  test_quantization.py
  test_roundtrip_en.py
  test_roundtrip_zh.py
```

## Design notes

- The packet is encoded as `fixed-size header + explicit-length body`.
- The core codec uses integer interval refinement instead of floating-point comparisons.
- The decoder fails closed on prompt drift, seed drift, token drift, or config mismatch inside the secret-bearing prefix.
- After the packet-bearing prefix resolves, the encoder can append a non-coded natural tail and the decoder ignores it.
- Retry-on-low-entropy is operational safety logic for the sender, not part of the shared decode fingerprint.
- The toy backend is the fastest way to validate protocol behavior; the Qwen `llama.cpp` backend is the realistic local CPU path.

More protocol details are in [spec.md](spec.md). A CCS-style paper draft is in [paper/ccs2026/main.tex](paper/ccs2026/main.tex).
