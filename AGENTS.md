# Ghostext Agent Guide

## Project Mission

本项目要做的是一个基于大模型 next-token 分布的自然语言隐写工程 demo：

- 发送端利用语言模型在每一步生成时给出的离散分布，把密文通过算术编码/区间编码映射到 token 选择过程中。
- 接收端在完全相同的模型、tokenizer、prompt、seed、生成配置下重放这些分布，并从观测到的 token 序列中恢复密文。
- 第一优先级是可解码性，其次才是自然度与容量。

## Current Scope

当前版本的默认边界如下：

- `工程 demo`，不是生产级隐写系统。
- `本地开源小模型`，目标量级约 `1.5B~4B`，要求中英文支持良好，CPU 上也能较快推理。
- `被动旁观者` 威胁模型，不处理主动篡改、鲁棒传输、纠错恢复。
- 默认管线是 `先加密，再隐写`。
- `不默认加入纠错码`。
- 编码端与解码端必须共享完全一致的：
  - 模型权重与版本
  - tokenizer
  - prompt / system prompt / 模板
  - seed
  - 温度、截断、最大步数等生成配置
  - 分布量化与区间编码实现

## Non-Goals

以下内容默认不属于 MVP：

- 对抗专门的隐写检测器
- 对抗文本被编辑、截断、改写后的鲁棒恢复
- 兼容只返回 top-logprobs 的闭源在线 API
- 在不同模型版本之间迁移解码
- 追求理论最优容量

## Agent Priorities

当你为本项目工作时，按下面的顺序做决策：

1. 保证编码端和解码端的确定性一致。
2. 保证端到端可恢复明文。
3. 保持协议、实现、测试三者一致。
4. 在不破坏前 3 条的前提下，再优化自然度、速度和容量。

如果某个改动能提升容量，但会明显增加解码漂移风险，默认不要做。

## Hard Rules

### 1. Determinism First

任何会影响 next-token 分布或候选集的逻辑都必须被视为协议的一部分，包括但不限于：

- logits 后处理
- 特殊 token mask
- 候选集裁剪策略
- 概率归一化
- 浮点到整数频数的量化
- 区间编码精度、进位、renormalization 规则
- 终止条件

这些内容一旦改动，必须同步更新 `spec.md` 和相关测试。

### 2. Fail Closed, Not Open

遇到配置不一致时，不要“尽量解码”，而要显式报错。至少要校验：

- `model_id`
- `tokenizer_hash`
- `prompt_template_id`
- `seed`
- `dtype / backend` 中的关键可复现项
- `candidate_policy_version`
- `codec_version`

### 3. Integer Arithmetic for Codec

区间/算术编码核心逻辑必须优先使用规范化的整数算术，不要把浮点比较留在编码器和解码器的关键路径里。

### 4. Encryption Is Mandatory

默认把用户明文先加密，再送入隐写编码。不要把“原始 UTF-8 明文直接嵌入文本”当作默认实现。

### 5. Bilingual Support Is Required

任何核心功能至少要考虑：

- 中文 prompt / 中文输出
- 英文 prompt / 英文输出

如果一个改动只在英文可用，必须明确标注。

## Expected Workflow

### Before Making Changes

- 先读 `spec.md`。
- 明确这次改动属于哪一层：
  - `crypto`
  - `packet framing`
  - `candidate policy`
  - `distribution quantization`
  - `range/arithmetic codec`
  - `model backend`
  - `cli / eval`

### While Implementing

- 优先做最小可验证改动。
- 优先写纯函数、可序列化配置、可回放中间状态。
- 如果实现中发现 `spec.md` 不完整，先补规范，再继续写代码。

### After Implementing

至少验证：

- 同配置 round-trip 成功
- 中文 case 成功
- 英文 case 成功
- 错误配置下能明确失败
- 修改没有破坏既有 packet/codec 行为

## Suggested Code Boundaries

除非用户明确要求别的结构，否则优先把职责拆成下面几层：

- `config`: 会影响复现与协议的结构化配置
- `crypto`: 密钥派生、加密解密、packet framing
- `model_backend`: 本地模型推理与 logits 提取
- `candidate_policy`: 候选 token 选择与过滤
- `quantization`: 概率到整数频数的规范量化
- `codec`: 区间/算术编码与镜像解码
- `pipeline`: `encode_secret_to_text` / `decode_text_to_secret`
- `eval`: round-trip、容量、自然度与耗时评测

不要把这些职责揉进一个巨型脚本里。

## Testing Expectations

至少应有以下测试：

- 固定 toy distribution 的 codec 单元测试
- packet 打包/解包测试
- 同一模型配置下的端到端 round-trip 测试
- 中文/英文 smoke tests
- 配置漂移导致失败的测试
- 候选集量化的稳定性测试

如果一次提交改动了 `candidate_policy`、`quantization` 或 `codec`，没有测试通常是不够的。

## Milestone Commit Policy

每个里程碑完成时，都应满足下面的收尾条件：

- 该里程碑对应的测试已经通过
- 文档与实现保持一致
- 立即创建一次 git commit，作为该里程碑的稳定检查点
- 立即将该检查点 git push 到约定远端

不要把多个已经通过测试的里程碑长期堆在同一个未提交、未推送的工作区里。

## Documentation Expectations

以下情况必须更新 `spec.md`：

- packet 格式变化
- 区间编码规则变化
- 候选集策略变化
- 停止条件变化
- 默认加密算法或 KDF 变化
- 评测指标变化

如果只是实现细节重构，但协议不变，可以只补充注释或开发说明。

## Decision Heuristics

当你不确定怎么选时，使用下面的默认原则：

- 选择更确定的，而不是更“聪明”的。
- 选择更容易回归测试的，而不是更难解释的。
- 选择更容易中英文统一的，而不是只对单语有效的。
- 选择更少隐式状态的，而不是依赖大量运行时副作用的。

一句话总结：这个仓库首先是一个“可稳定复现的隐写编码实验台”，然后才是一个“尽量自然的文本生成器”。
