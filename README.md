# HideText-codex

HideText 是一个“把秘密消息藏进自然语言生成过程里”的工程 demo。它不在事后改写文本，而是把密文映射到每一步 next-token 分布对应的离散选择里。

当前仓库已经实现：

- 固定协议配置的 packet framing
- `scrypt + ChaCha20-Poly1305` 的 `encrypt-then-stego`
- 确定性的候选集裁剪与整数频数量化
- 一个纯整数的双阶段 finite-interval codec
- 可复现的双语 toy backend
- 一个基于 `llama.cpp` 的真实 Qwen GGUF backend
- `encode / decode / eval` CLI
- 中英文 round-trip、负例、CLI 自动测试
- 可选的真实 Qwen CPU 集成测试

默认的快速测试仍然主要依赖 `ToyCharBackend`，因为它能稳定、快速地验证协议逻辑；与此同时，仓库现在也接入了一个真实本地模型后端：`Qwen/Qwen3-4B-Instruct-2507` 的 GGUF 量化版本，可通过 `llama.cpp` 在 CPU 上运行。

## Quick Start

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

如果你要跑真实模型后端，先安装可选依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[llm]'
```

然后下载一个适合 CPU 的 GGUF。当前验证过的是：

- 模型仓库：`bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`
- 文件名：`Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf`

中文 demo：

```bash
PYTHONPATH=src python3 -m hidetext.cli eval \
  --prompt '请写一段温柔自然的中文短文。' \
  --passphrase pass-zh \
  --message '今晚七点在老地方见。' \
  --seed 17
```

英文 demo：

```bash
PYTHONPATH=src python3 -m hidetext.cli eval \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase pass-en \
  --message 'Meet me near the station at seven.' \
  --seed 29
```

真实 Qwen CPU demo：

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
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase qwen-real-test \
  --message '真实模型联调成功' \
  --seed 7
```

如果你希望从文件读取 `prompt`、`passphrase` 或 `seed`，CLI 现在也支持：

```bash
PYTHONPATH=src python3 -m hidetext.cli eval \
  --prompt-file prompts/zh.txt \
  --passphrase-file secrets/passphrase.txt \
  --seed-file secrets/seed.txt \
  --message '今晚七点在老地方见。'
```

其中：

- `--prompt-file` 读取 UTF-8 文本作为 prompt
- `--passphrase-file` 读取 UTF-8 文本作为口令
- `--seed-file` 读取文本里的整数作为 seed
- 文件末尾多余的换行会被自动去掉
- 真实模型后端还支持 `--backend llama-cpp --model-path ...`
- 候选策略和 codec 关键参数可以直接通过 CLI 调整：
  `--top-p`、`--max-candidates`、`--min-entropy-bits`、`--totfreq`、
  `--header-token-budget`、`--body-token-budget`

如果你想分别编码和解码：

```bash
PYTHONPATH=src python3 -m hidetext.cli encode \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase demo \
  --message 'CLI roundtrip works.' \
  --seed 11
```

```bash
PYTHONPATH=src python3 -m hidetext.cli decode \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase demo \
  --text '<encode 返回的 text 字段>' \
  --seed 11
```

## Repo Layout

```text
src/hidetext/
  config.py
  packet.py
  crypto.py
  model_backend.py
  llama_cpp_backend.py
  candidate_policy.py
  quantization.py
  codec.py
  pipeline.py
  encoder.py
  decoder.py
  cli.py
tests/
  test_packet.py
  test_crypto.py
  test_quantization.py
  test_codec_toy.py
  test_roundtrip_zh.py
  test_roundtrip_en.py
  test_failures.py
  test_cli.py
  test_llama_cpp_integration.py
```

## Current Design Notes

- packet 是 `fixed header + body` 两阶段编码，不直接一次性编码整包
- header 中包含运行时配置指纹，用于 fail-closed 校验
- codec 使用精确整数区间收缩，不走浮点比较
- 当前 demo 更偏向“稳定可解码”，不追求高容量或抗检测
- 真实 Qwen backend 目前使用固定的 Qwen ChatML 风格 user->assistant prompt 模板
- 真实模型 smoke test 为了可复现，会固定 salt / nonce；正常 CLI 仍默认随机生成它们

真实模型集成测试命令：

```bash
HIDETEXT_LLAMA_MODEL_PATH=/abs/path/to/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
HIDETEXT_LLAMA_THREADS=8 \
HIDETEXT_LLAMA_CTX=4096 \
HIDETEXT_LLAMA_BATCH=128 \
.venv/bin/python -m unittest tests.test_llama_cpp_integration -v
```

更多协议细节见 [spec.md](spec.md) ，agent 约束见 [AGENTS.md](AGENTS.md)。
