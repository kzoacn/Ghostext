# HideText-codex 中文说明

[English README](README.md)

HideText 是一个确定性的文本隐写 demo。它会先加密消息，再把密文映射到语言模型生成时的 token 选择里；只要发送端和接收端共享同一模型、prompt、口令和 seed，就可以把消息恢复出来。

现在默认的用户路径已经切到真实本地 `llama.cpp` 后端。`toy` backend 仍然保留，但主要用于测试和协议调试。

## 快速开始

先安装带本地 LLM 支持的版本：

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[llm]'
```

用一条命令先跑通一次 encode/decode round-trip：

```bash
.venv/bin/hidetext encode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase demo-pass \
  --message '今晚七点在河边老地方见。' \
| .venv/bin/hidetext decode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase demo-pass
```

第一次运行时，HideText 会按下面的顺序找模型：

- 先看你有没有传 `--model-path`
- 否则看 `HIDETEXT_MODEL_PATH` 或 `HIDETEXT_LLAMA_MODEL_PATH`
- 如果还没有，就自动下载默认 GGUF 到 `~/.cache/hidetext/models/qwen35-2b-q4ks/`

当前默认模型是：

- model id: `Qwen/Qwen3.5-2B`
- repo: `bartowski/Qwen_Qwen3.5-2B-GGUF`
- file: `Qwen_Qwen3.5-2B-Q4_K_S.gguf`

## 日常使用

平时 `encode` 基本只需要这几个参数：

```bash
.venv/bin/hidetext encode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase 'river-pass' \
  --message '今晚七点在河边老地方见。'
```

`decode` 默认会从标准输入读取隐写文本，所以最短路径通常是直接走管道：

```bash
.venv/bin/hidetext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| .venv/bin/hidetext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想把生成出的文本先保存下来：

```bash
.venv/bin/hidetext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
  > stego.txt

.venv/bin/hidetext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --text-file stego.txt
```

## 你通常不再需要操心的参数

CLI 现在内置了一套真实模型预设，所以多数情况下你不需要再手动调这些：

- `--seed`
- `--ctx-size`
- `--batch-size`
- `--top-p`
- `--max-candidates`
- `--min-entropy-bits`
- `--totfreq`

当前默认预设是：

- backend: `llama-cpp`
- seed: `7`
- ctx size: `4096`
- batch size: `128`
- top-p: `0.995`
- max candidates: `64`
- min entropy bits: `0.0`
- total frequency: `4096`

也就是说，如果你不显式覆盖，CLI 就会直接用这组值。

## 常见覆盖方式

如果你已经有自己的 GGUF 模型：

```bash
.venv/bin/hidetext encode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| .venv/bin/hidetext decode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想把默认下载目录换掉：

```bash
HIDETEXT_MODEL_DIR=/data/hidetext-models \
.venv/bin/hidetext encode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| HIDETEXT_MODEL_DIR=/data/hidetext-models \
  .venv/bin/hidetext decode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想看结构化元数据输出：

```bash
.venv/bin/hidetext encode \
  --json \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

如果你需要完全安静输出，可以加 `--quiet`：

```bash
.venv/bin/hidetext encode \
  --quiet \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

## Toy backend

`ToyCharBackend` 还在，但默认不是给普通使用者准备的，而是给测试和协议实验准备的：

```bash
.venv/bin/hidetext encode \
  --backend toy \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase test-pass \
  --message 'toy backend roundtrip' \
| .venv/bin/hidetext decode \
  --backend toy \
  --prompt 'Write a calm and readable English paragraph.' \
  --passphrase test-pass
```

## 测试

运行当前测试集：

```bash
env PYTHONPATH=src python3 -m unittest discover -s tests -v
```

真实模型 smoke test 仍然保持手动开启，因为它需要本地 GGUF：

```bash
HIDETEXT_LLAMA_MODEL_PATH=/abs/path/to/model.gguf \
env PYTHONPATH=src python3 -m unittest tests.test_llama_cpp_integration -v
```

## 深入说明

如果你想看协议级的设计和实现细节，请读 [spec.md](spec.md)。README 现在刻意偏向“先让使用者快速跑起来”。
