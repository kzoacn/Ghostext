# HideText 规格说明

## 1. 项目目标

HideText 是一个基于大模型 next-token 分布的自然语言隐写工程 demo。它的核心目标是：

- 把一段秘密消息嵌入到自然语言生成过程中，而不是事后修改文本。
- 利用语言模型每一步给出的离散分布作为信道。
- 使用算术编码 / 区间编码把密文映射为 token 选择。
- 在编码端和解码端共享完全相同配置的前提下，实现稳定可恢复。

本项目第一版优先级如下：

1. 可解码性
2. 工程可复现性
3. 中英文兼容
4. 自然度
5. 容量

## 2. 非目标

第一版默认不解决以下问题：

- 对抗专门的隐写检测器
- 文本被改写、删减、错别字污染后的鲁棒恢复
- 多模型或模型版本漂移下的兼容解码
- 只依赖在线闭源 API 的实现
- 纠错码与对抗信道编码

## 3. 威胁模型

默认威胁模型是 `被动旁观者`：

- 旁观者可以看到生成出的文本。
- 旁观者不知道发送端与接收端共享的配置和密钥。
- 旁观者不会主动篡改文本，也不会插入、删除或重排 token。

这意味着 MVP 的重点是：

- 在共享配置下正确恢复消息
- 生成文本尽量自然

而不是：

- 抗主动攻击
- 抗编辑
- 抗专业检测器

## 4. 核心假设

编码端和解码端必须共享完全一致的上下文与运行条件：

- 相同模型权重
- 相同 tokenizer
- 相同 prompt 与模板
- 相同 seed
- 相同生成配置
- 相同特殊 token 处理规则
- 相同候选集裁剪规则
- 相同概率量化规则
- 相同区间编码精度与 renormalization 规则

如果上述任一条件不一致，系统应当显式失败，而不是尝试“近似解码”。

## 5. 术语

- `D_t`：第 `t` 步的 next-token 分布。
- `candidate set`：在 `D_t` 上经过 mask、裁剪、重归一化后可参与隐写的 token 集合。
- `quantized CDF`：把候选 token 的概率转成整数频数后得到的累计分布。
- `stego packet`：送入隐写编码的最终二进制载荷，默认已加密。
- `sender`：编码端，把密文嵌入生成文本。
- `receiver`：解码端，从文本中恢复密文。

## 6. 系统概览

系统包含 6 个主要层次：

1. `crypto layer`
   - 把用户明文变成加密后的二进制 packet
2. `model backend`
   - 对本地模型做推理并提取 logits / logprobs
3. `candidate policy`
   - 决定哪些 token 可参与编码
4. `quantization layer`
   - 把浮点概率稳定地量化为整数频数
5. `range/arithmetic codec`
   - 利用每一步的离散分布把 packet 映射到 token 选择
6. `pipeline + cli`
   - 为用户暴露 encode / decode / eval 接口

## 7. 默认运行环境

第一版默认面向：

- 本地开源小模型
- 参数规模约 `1.5B~4B`
- 中英文支持良好
- CPU 上也能较快推理

实现上应优先支持：

- 单机本地加载
- 稳定获得全词表 logits
- 可锁定 tokenizer 与模型版本

## 8. 默认使用场景

### 8.1 编码

输入：

- 一段秘密消息
- 一个共享密钥或口令
- 一个 cover prompt
- 一组共享配置

输出：

- 一段看起来自然的文本

### 8.2 解码

输入：

- 生成出的文本
- 相同的共享密钥或口令
- 相同的 cover prompt
- 相同的共享配置

输出：

- 恢复出的秘密消息

## 9. 默认配置原则

默认遵守下列原则：

- `先加密，再隐写`
- `不默认使用纠错码`
- `seed 是配置的一部分`
- `所有协议敏感逻辑都必须可序列化`
- `默认为了稳定性牺牲一部分容量`

## 10. 数据流

默认数据流如下：

1. 用户明文 `plaintext`
2. UTF-8 编码为字节串
3. 使用共享密钥进行 AEAD 加密
4. 形成 `stego packet`
5. 将 `stego packet` 转成 bitstream
6. 编码器在每一步读取模型分布 `D_t`
7. 候选集裁剪并量化为整数 CDF
8. 区间编码器据此选择 token
9. 得到输出文本

解码时执行逆过程。

## 11. Packet 设计

第一版建议使用显式长度头，以便接收端知道何时停止。

当前实现采用固定长度头部，结构如下：

```text
magic[4] | version[1] | flags[1] | kdf_id[1] | aead_id[1] | salt_len[1] | nonce_len[1] | ct_len[4] | config_fingerprint[8] | salt[...] | nonce[...] | ciphertext_and_tag[ct_len]
```

说明：

- `magic`：用于快速检测配置错误或恢复失败
- `version`：协议版本
- `flags`：保留位
- `kdf_id`：密钥派生算法编号
- `aead_id`：AEAD 算法编号
- `salt_len`：KDF salt 长度
- `nonce_len`：nonce 长度
- `ct_len`：`ciphertext || tag` 的总长度
- `config_fingerprint`：运行时协议配置、模型元数据与 prompt 的 64-bit 指纹
- `salt`：KDF salt
- `nonce`：AEAD nonce
- `ciphertext_and_tag`：AEAD 输出

第一版可选的默认密码学配置：

- `KDF`: `scrypt` 或 `Argon2id`
- `AEAD`: `ChaCha20-Poly1305` 或 `AES-256-GCM`

当前实现固定默认组合：

- `KDF`: `scrypt`
- `AEAD`: `ChaCha20-Poly1305`

`header` 本身作为 AEAD 的 associated data 参与认证。

## 12. Cover Text 与 Prompt

第一版采用 `共享 cover prompt` 模式：

- 编码端和解码端共享相同 prompt
- prompt 是协议输入的一部分
- prompt 不需要对外保密，但必须完全一致

建议 prompt 约束输出为一段短文本，例如：

- 中文短段落
- 英文短段落
- 问答式回复
- 指定主题的小短文

第一版不要求自动规划复杂篇章结构。

## 13. 候选集策略

候选集策略是协议的一部分，必须完全确定化。

### 13.1 基本流程

对第 `t` 步的分布 `D_t`：

1. 先屏蔽不允许的特殊 token
2. 根据固定规则裁剪候选集
3. 对候选集重新归一化
4. 计算该步熵 `H_t`
5. 决定该步是否允许嵌入消息

### 13.2 默认建议

第一版建议采用以下保守策略：

- 先按概率降序排序，tie-break 使用 token id
- 使用 `top-p` 截断，例如 `p = 0.95 ~ 0.98`
- 同时设置 `max_candidates`，例如 `32` 或 `64`
- 若候选集大小小于 `2`，则该步不编码
- 若 `H_t < H_min`，则该步不编码

### 13.3 非编码步

如果某一步不编码消息，则发送端与接收端都应使用相同的确定性规则选 token，例如：

- 选择候选集中的最高概率 token

这样做的目的：

- 保持上下文同步
- 避免低熵位置引入不稳定选择

## 14. 概率量化

为了保证可复现性，浮点概率不能直接进入 codec 核心逻辑，必须先量化成整数频数。

### 14.1 目标

给定候选 token 概率 `p_i`，构造整数频数 `f_i`，满足：

- `f_i >= 1`
- `sum(f_i) = TOTFREQ`
- 量化过程完全确定

### 14.2 默认建议

- `TOTFREQ = 2^16`
- 先计算 `raw_i = p_i * TOTFREQ`
- 使用固定的向下取整 + remainder 分发方案
- 对所有 tie 使用固定顺序打破，例如 token id

### 14.3 注意事项

以下逻辑都必须写入实现规范：

- 当 `sum(f_i)` 超过 `TOTFREQ` 时如何回退
- 当 `sum(f_i)` 小于 `TOTFREQ` 时如何补齐
- 候选集过大导致无法保证 `f_i >= 1` 时如何裁剪

## 15. 区间编码 / 算术编码设计

本项目的核心思想是：

- 把密文 bitstream 看作区间编码器的输入
- 把每一步量化后的候选 token 分布看作当前步的符号模型
- 根据 bitstream 当前落点选择一个 token
- 再用该 token 对应的子区间更新编码状态

### 15.1 推荐实现方式

第一版推荐使用 `整数 range coder / arithmetic coder`，而不是浮点区间实现。

关键要求：

- 固定整数精度，例如 `64-bit`
- 固定 renormalization 规则
- 固定 bit 输出/消费顺序
- 固定结束条件

当前 MVP 实现使用的是一个更直接的 `finite-message interval codec`：

- 对一个长度为 `n` 字节的 segment，把消息看成区间 `0..2^(8n)-1` 上的一个整数点
- 每一步根据量化后的候选分布，把当前整数区间按 CDF 划分为若干子区间
- 发送端选择包含目标整数点的那个 token
- 接收端观察 token 后，用同样的子区间规则收缩区间
- 当区间宽度收缩到 `1` 时，该 segment 被唯一恢复

这个实现仍然完全使用整数算术，但不依赖固定宽度 renormalization；它更适合作为工程 demo 的首版可验证原型。

### 15.2 发送端与接收端的对偶关系

当前实现按以下对偶关系工作：

- 发送端：把 segment 字节串映射为一个目标整数，并在每一步按 token 分布收缩区间、选出包含该整数的 token。
- 接收端：观察到 token 序列后，在相同的分布序列上执行镜像区间收缩，直到区间宽度为 `1`，再恢复对应字节串。

这两个过程必须使用同一个 codec 规范。

### 15.3 终止条件

当前实现采用 `固定头 segment + 显式长度体 segment` 的两阶段停止策略：

- 先把固定长度 packet 头编码为第一个 segment
- 接收端在头 segment 区间宽度变为 `1` 后即可恢复 header
- header 中给出 `salt_len`、`nonce_len` 和 `ct_len`，从而确定 body segment 的精确长度
- 再对 body segment 执行同样的区间编码
- 发送端在两个 segment 都收敛后停止生成
- 接收端在 body segment 完整恢复后停止解码

这样可以避免在 packet 总长度未知时直接对整包做一次性编码。

## 16. 推荐默认参数

这些参数是建议值，不是强制常数：

```text
temperature = 1.0
top_p = 0.95 或 0.98
max_candidates = 32 或 64
min_entropy_bits = 1.0 ~ 1.5
totfreq = 65536
range_precision_bits = 64
```

设计原则：

- 先保证稳定性
- 再逐步向更高容量推进

## 17. 失败模式

第一版必须显式考虑以下失败模式：

1. 模型或 tokenizer 不一致
2. prompt 不一致
3. seed 或生成配置不一致
4. 特殊 token mask 不一致
5. 候选集排序或 tie-break 不一致
6. 概率量化逻辑不一致
7. 区间编码 renormalization 不一致
8. 输出文本被额外改动
9. 解码得到的 packet 无法通过 AEAD 校验

默认处理方式：

- 尽早报错
- 不做“模糊恢复”

## 18. 评测指标

第一版至少评测以下指标：

### 18.1 正确性

- `round-trip success rate`
- `AEAD verify success rate`
- 不同消息长度下的恢复成功率

### 18.2 容量

- `bits per token`
- `bits per character`
- `embedded_bits / sum(H_t)` 的熵利用率

### 18.3 文本质量

- 平均 token logprob
- 相对基线 perplexity 或 pseudo-perplexity
- 人工主观可读性检查

### 18.4 工程性能

- CPU 下编码耗时
- CPU 下解码耗时
- tokens/sec

## 19. 测试计划

至少需要以下测试层次：

### 19.1 单元测试

- toy 离散分布上的 codec 正反例
- 量化前后频数和 CDF 合法性
- packet 头编解码
- 加密与解密

### 19.2 集成测试

- 固定 prompt + 固定模型 + 固定 seed 的 round-trip
- 中文消息 round-trip
- 英文消息 round-trip
- 不同长度消息 round-trip

### 19.3 负例测试

- 故意换 seed
- 故意换 prompt
- 故意换模型
- 故意改一处 token

应观察到：

- 解码失败或校验失败
- 错误信息足够明确

## 20. 建议的仓库结构

第一版建议采用如下结构：

```text
src/hidetext/
  config.py
  packet.py
  crypto.py
  model_backend.py
  candidate_policy.py
  quantization.py
  codec.py
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
```

## 21. MVP 里程碑

里程碑的完成标准不仅是“功能看起来可用”，还包括：

- 该里程碑相关测试通过
- 关键文档已同步更新
- 完成后立即进行一次 git commit，形成明确检查点
- 完成后立即将该检查点 git push 到远端

### M0: 规范与玩具原型

- 明确 packet 格式
- 明确候选集策略
- 明确量化规则
- 用 toy distribution 跑通 codec

### M1: 单模型端到端打通

- 接上本地小模型
- 固定 prompt、固定 seed
- 能完成基础 round-trip

### M2: 中英文双语 demo

- 中文 prompt 与消息成功
- 英文 prompt 与消息成功
- 提供 CLI 示例

### M3: 评测与展示

- 输出成功率、bits/token、耗时
- 给出若干自然文本示例

## 22. 后续扩展方向

这些方向不属于 MVP，但值得保留：

- 对抗检测器的自然度优化
- 句子级或语义级规划
- 自适应容量分配
- 鲁棒恢复与纠错码
- 对 paraphrase / edit 的抗性
- 多模型后端统一接口

## 23. 一句话总结

HideText 的第一版不是在追求“最隐蔽”或“最大容量”，而是在追求一个可稳定复现、可中英文演示、基于本地小模型的 next-token 分布隐写工程 demo。凡是和这个目标冲突的优化，默认都应让位于可解码性与确定性。
