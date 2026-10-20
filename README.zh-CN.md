# Ghostext 中文说明

S&P 2027 研究修订采用 `guarded-prefix-v2` 编码：给每段加入独立随机后缀，
在有效载荷前缀已经确定时停止，无需把整个整数区间缩小到一个点。
这修复了旧编码器在段末产生的一类可测量采样偏差。双方必须使用相同修订和
`--guard-bits` 参数（默认 128）；`--guard-bits 0` 仅用于旧协议对照实验。
该修复不等于已经证明抗文本检测、长度分析或重试行为分析。
认证保护的是解码后的数据包。只改变被丢弃的保护位或尾随文本的可见编辑，
可能仍会恢复同一明文并通过认证。

实验选项 `--empty-candidate-search 4096` 会在原候选池没有稳定 token 时，
搜索更大的有序候选池。该选项默认关闭，双方必须设置相同数值。
它会改变候选分布；历史结果及严格候选策略的实验均未启用它。

[English README](README.md)

Ghostext 是一个文本隐写 demo。它会先加密消息，再把密文映射到语言模型生成时的 token 选择里；只要发送端和接收端共享同一模型、prompt、口令和 seed，就可以把消息恢复出来。

现在默认的用户路径已经切到真实本地 `llama.cpp` 后端。

## 快速开始

本研究修订需要从 `sp2027-revision` 分支安装，PyPI 上的旧发布不包含这些改动：

```bash
git clone --branch sp2027-revision https://github.com/kzoacn/Ghostext.git
cd Ghostext
python -m pip install -e '.[llm]'
```

用一条命令先跑通一次 encode/decode round-trip：

```bash
ghostext encode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase demo-pass \
  --message '今晚七点在河边老地方见。' \
| ghostext decode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase demo-pass
```

查看完整命令帮助：

```bash
ghostext --help
ghostext encode --help
ghostext decode --help
ghostext benchmark --help
```

第一次运行时，Ghostext 会按下面的顺序找模型：

- 先看你有没有传 `--model-path`
- 否则看 `GHOSTEXT_MODEL_PATH` 或 `GHOSTEXT_LLAMA_MODEL_PATH`
- 如果还没有，就自动下载默认 GGUF 到 `~/.cache/ghostext/models/qwen35-2b-q4ks/`

当前默认模型是：

- model id: `Qwen/Qwen3.5-2B`
- repo: `bartowski/Qwen_Qwen3.5-2B-GGUF`
- file: `Qwen_Qwen3.5-2B-Q4_K_S.gguf`

## 日常使用

平时 `encode` 基本只需要这几个参数：

```bash
ghostext encode \
  --prompt '请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。' \
  --passphrase 'river-pass' \
  --message '今晚七点在河边老地方见。'
```

`decode` 默认会从标准输入读取隐写文本，所以最短路径通常是直接走管道：

```bash
ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| ghostext decode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想把生成出的文本先保存下来：

```bash
ghostext encode \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
  > stego.txt

ghostext decode \
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
ghostext encode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| ghostext decode \
  --model-path /abs/path/to/model.gguf \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想把默认下载目录换掉：

```bash
GHOSTEXT_MODEL_DIR=/data/ghostext-models \
ghostext encode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.' \
| GHOSTEXT_MODEL_DIR=/data/ghostext-models \
  ghostext decode \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass
```

如果你想看结构化元数据输出：

```bash
ghostext encode \
  --json \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

如果你需要完全安静输出，可以加 `--quiet`：

```bash
ghostext encode \
  --quiet \
  --prompt 'Write a short paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Meet near the riverside at seven.'
```

## 简单 benchmark

可以用 `benchmark` 子命令快速测这四个指标：

- `encode latency`
- `decode latency`
- `encode bits/token`
- `ppl`

使用本地 LLM 后端（`llama-cpp`），并沿用 README 开头示例里的英文 `prompt/passphrase/message`：

```bash
ghostext benchmark \
  --backend llama-cpp \
  --prompt 'Write a short, natural paragraph about a quiet evening walk.' \
  --passphrase demo-pass \
  --message 'Attack at Dawn!' \
  --runs 3 \
  --json
```

## 深入说明

如果你想看协议级的设计和实现细节，请读 [spec.md](spec.md)。README 现在刻意偏向“先让使用者快速跑起来”。
