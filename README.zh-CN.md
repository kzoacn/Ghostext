# HideText-codex 中文说明

[English README](README.md)

HideText 是一个确定性的文本隐写 demo：它不是先生成自然语言再做后处理，而是直接把加密后的秘密载荷映射到语言模型每一步的 next-token 选择里。在发送端和接收端共享同一模型、tokenizer、prompt、seed 和 codec 配置的前提下，接收端可以镜像重放这些分布并恢复明文。

这个仓库优先保证的是可复现和可解码，然后才是自然度与容量。

## 已实现能力

- 带运行时配置指纹的固定 packet framing
- `scrypt + ChaCha20-Poly1305` 的 encrypt-then-stego 载荷构造
- 确定性的候选集裁剪与整数频数量化
- 整数 finite-interval header/body codec
- 用于快速协议验证的双语 toy backend
- 基于 `llama.cpp` 的本地 Qwen GGUF CPU 后端
- `encode` / `decode` / `eval` CLI
- 对 prompt 漂移、seed 漂移、文本改动、重试耗尽的 fail-closed 测试
- 低熵监测与自动重试
- secret-bearing prefix 完成后的自然尾部生成

## 快速开始

创建本地虚拟环境并安装基础依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

运行默认测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 真实模型环境

安装可选的本地 LLM 依赖：

```bash
.venv/bin/python -m pip install -e '.[llm]'
```

下载当前验证过的 GGUF 模型：

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

运行真实模型 smoke test：

```bash
HIDETEXT_LLAMA_MODEL_PATH=/abs/path/to/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
HIDETEXT_LLAMA_THREADS=8 \
HIDETEXT_LLAMA_CTX=4096 \
HIDETEXT_LLAMA_BATCH=128 \
.venv/bin/python -m unittest tests.test_llama_cpp_integration -v
```

当前仓库已经在本地完成过：

- toy backend 的完整默认测试
- `Qwen/Qwen3-4B-Instruct-2507` GGUF `Q4_K_M` 的 CPU round-trip smoke test

## 低熵自动重试

当模型进入近乎确定性的低熵区域时，候选集会退化到几乎无法承载信息，极端情况下只有一个 token，消息就可能编码不进去。HideText 现在默认会这样处理：

- 监测连续 `32` 个 token step 的滚动窗口
- 如果这个窗口上的平均候选熵低于 `0.1` bit，就放弃当前尝试
- 用新的随机 salt / nonce 重新生成 packet，再做一次编码
- 如果连续 `3` 次都失败，就显式报错，并建议更换 cover prompt 或减少消息长度

这些参数都是发送端本地运行时安全策略，不进入 packet fingerprint：

- `--low-entropy-window-tokens`
- `--low-entropy-threshold-bits`
- `--max-encode-attempts`
- `--stall-patience-tokens`

如果你在做固定 packet 的受控实验，可以传 `--low-entropy-window-tokens 0` 显式关闭这个 detector。

## 自然结束

HideText 现在不必在 packet 刚好可解码的那个 token 上硬停。只要 header 和 body 都已经编码完成：

- 承载秘密的前缀已经完整，可被解码
- 发送端可以继续随机采样一小段尾部，让文本结尾更自然
- 这段尾部不参与 codec，解码端会直接忽略
- 解码端在 packet 完整恢复后就停止，不要求文本也在那个位置结束

可以用 `--natural-tail-max-tokens` 限制这段尾部的最大长度；如果传 `0`，就会回到“编码完成立即停止”的旧行为。

## CLI 用法

toy backend round-trip：

```bash
.venv/bin/python -m hidetext.cli eval \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase pass-en \
  --message 'Meet me near the station at seven.' \
  --seed 29
```

真实 Qwen CPU round-trip：

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
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase real-pass \
  --message '今晚七点在河边老地方见。' \
  --seed 7
```

如果希望从文件读取输入：

```bash
.venv/bin/python -m hidetext.cli eval \
  --prompt-file prompts/zh.txt \
  --passphrase-file secrets/passphrase.txt \
  --seed-file secrets/seed.txt \
  --message '今晚七点在老地方见。'
```

## 示例

### 英文示例

- Backend: `toy`
- Prompt / cover prompt: `Write a calm and readable English paragraph about a quiet evening walk.`
- Secret message: `Meet near the riverside at seven.`
- Passphrase: `demo-en`
- Seed: `29`

命令：

```bash
.venv/bin/python -m hidetext.cli encode \
  --prompt 'Write a calm and readable English paragraph about a quiet evening walk.' \
  --passphrase demo-en \
  --message 'Meet near the riverside at seven.' \
  --seed 29
```

隐写文本片段：

```text
The afternoon feels cllm, and the corner cafe keeps a warm light on while people talk softlv along the street. A quiet breeze moves past the window, the pages stay hflf open on tte table, and the room keeps a steady brightness.
```

### 中文示例

- Backend: `toy`
- Prompt / cover prompt: `请写一段自然、柔和、连贯的中文短文，描写傍晚散步时看到的街景。`
- Secret message: `今晚七点在河边老地方见。`
- Passphrase: `demo-zh`
- Seed: `17`

命令：

```bash
.venv/bin/python -m hidetext.cli encode \
  --prompt '请写一段自然、柔和、连贯的中文短文，描写傍晚散步时看到的街景。' \
  --passphrase demo-zh \
  --message '今晚七点在河边老地方见。' \
  --seed 17
```

隐写文本片段：

```text
今天的风很轻，街角的咖啡店还亮着暖黄的静，路过的人慢慢聊着天，城天显得安静而柔和。午后的窗边有一轻淡淡香茶的，桌上书书翻到一半，雨声落在树叶上，房间里有种稳稳的明亮。
```

## 仓库结构

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

## 设计说明

- packet 采用 `固定头 + 显式长度 body` 两阶段编码
- 核心 codec 使用整数区间收缩，不依赖浮点比较
- 解码器在秘密前缀内部遇到 prompt、seed、token 或配置漂移时都会 fail closed
- packet 前缀恢复后，编码器可以继续追加不参与编解码的自然尾部，解码器会忽略它
- 低熵重试属于发送端运行时安全策略，不属于共享解码 fingerprint
- toy backend 适合快速验证协议；Qwen `llama.cpp` backend 对应真实本地 CPU 路径

更完整的协议细节见 [spec.md](spec.md)。CCS 风格的论文草稿见 [paper/ccs2026/main.tex](paper/ccs2026/main.tex)。
