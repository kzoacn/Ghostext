# HideText-codex

HideText 是一个“把秘密消息藏进自然语言生成过程里”的工程 demo。它不在事后改写文本，而是把密文映射到每一步 next-token 分布对应的离散选择里。

当前仓库已经实现：

- 固定协议配置的 packet framing
- `scrypt + ChaCha20-Poly1305` 的 `encrypt-then-stego`
- 确定性的候选集裁剪与整数频数量化
- 一个纯整数的双阶段 finite-interval codec
- 可复现的双语 toy backend
- `encode / decode / eval` CLI
- 中英文 round-trip、负例和 CLI 自动测试

当前仓库还没有接入真实开源 LLM。现在的 `ToyCharBackend` 主要用于把协议、镜像解码和工程流程先跑通；后续可以在现有 backend 接口上替换为真正的本地模型后端。

## Quick Start

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

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
```

## Current Design Notes

- packet 是 `fixed header + body` 两阶段编码，不直接一次性编码整包
- header 中包含运行时配置指纹，用于 fail-closed 校验
- codec 使用精确整数区间收缩，不走浮点比较
- 当前 demo 更偏向“稳定可解码”，不追求高容量或抗检测

更多协议细节见 [spec.md](spec.md) ，agent 约束见 [AGENTS.md](AGENTS.md)。
