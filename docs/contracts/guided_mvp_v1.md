# Guided MVP Contract v1

本文档定义第一版 Guided 英文音节对齐的稳定业务契约。权威机器可读配置为 `configs/contracts/guided_mvp_v1.yaml`，JSON 输出结构为 `schemas/guided_syllable_result_v1.schema.json`，跨字段语义由 `syllable_recognition.contracts.guided.validate_guided_result` 检查。

本契约已通过 Stage 1 门禁。Stage 2 已使用 `librispeech-english-v1`、`cmudict 1.1.1`、`g2p-en 2.1.0`、`arpabet-word-internal-v1` 和 MFA 3.3.7 在 12 条真实 LibriSpeech 样本上验证。契约固定的是产品边界，不表示 CTC 模型已经训练完成。

## 1. 范围

覆盖：

- 已知目标文本下的音节约束对齐；
- 目标音节的起止时间；
- phones、onset、nucleus、coda 和 stress；
- 对齐置信度；
- 插入、删除、重复和明显不匹配事件。

不覆盖：

- Open 自由语音识别；
- 发音正确率评分；
- VTL 轨迹生成；
- 严格流式推理；
- 未经训练的模型能力声明。

## 2. 输入

- `mode` 必须显式为 `guided`。
- 输入是一条单声道 16 kHz 英文波形。
- 必须提供非空目标文本。
- 时间单位统一为秒。
- 第一版允许近实时分块，但 HuBERT/Wav2Vec 2.0 是双向 encoder，不能称为严格流式。

## 3. 处理语义

目标链路：

```text
目标文本
  -> librispeech-english-v1 规范化
  -> CMUdict/OOV G2P
  -> 目标 ARPAbet 音素
  -> arpabet-word-internal-v1 音节化

学习者语音
  -> HuBERT/Wav2Vec Phone CTC logits

目标序列 + logits
  -> CTC 约束对齐
  -> 目标音节时间戳、置信度和事件
```

当前 MFA 仅用于离线训练伪标签生产，不属于最终 Guided 在线推理链路。

## 4. 音节定义

- 每个音节恰好有一个 ARPAbet 元音 nucleus。
- 只在单词内部音节化，不跨词重音节化。
- onset 和 coda 只包含辅音。
- nucleus 不含重音数字。
- stress 单独保存为 `0`、`1` 或 `2`。
- 公共接口使用空数组表示无 onset/coda；训练词表以后使用显式 `<empty>`。
- `phones` 必须严格等于 onset + 带重音 nucleus + coda。
- `label` 必须是 phones 使用 `-` 连接后的结果。

例如：

```text
onset=[HH], nucleus=AH, stress=0, coda=[]
-> phones=[HH, AH0]
-> label=HH-AH0
```

## 5. 输出语义

- `target_text` 是实际用于对齐的规范化文本。
- `segments[*].phones` 和 `label` 描述目标音节，不是 Open 模式声学预测。
- `start/end` 是声学约束对齐时间戳。
- `confidence` 是 CTC 对齐置信度，不是发音正确率。
- `token_id` 在 v1 中允许为 `null`。
- deletion 没有声学时间戳；其他事件必须有有效区间。
- failed alignment 不得包含伪造成功的 segments。
- segment 时间戳必须严格递增且不重叠。
- target index 必须严格递增，并与 deletion 事件共同覆盖目标序列。

参考有效输出为 `tests/fixtures/contracts/guided_valid.json`。

## 6. 数据划分

| split | 用途 |
| --- | --- |
| `train-clean-100` | 参数、词表和统计量拟合 |
| `dev-clean` | checkpoint、阈值和规则选择 |
| `test-clean` | 最终一次性报告 |

必须保留官方说话人划分，禁止按句子随机重分。test-clean 不得参与模型、阈值和规则选择。

## 7. 评价

- Phone CTC：PER。
- Guided：对齐成功率、时间戳 MAE、插入/删除/重复/不匹配计数。
- Boundary：`+/-20 ms` 和 `+/-50 ms` Precision/Recall/F1。
- Segmentation：漏切率、过切率、音节数量误差。
- System：实时率、端到端延迟、峰值显存和模型大小。

对齐置信度和发音正确率必须分开报告。

## 8. 门禁

Stage 1 已通过以下检查：

- 有效 fixture 通过 JSON Schema 和语义校验；
- 非法 phone 分解被拒绝；
- 重叠、非单调和越界时间戳被拒绝；
- `[0, 1]` 外置信度被拒绝；
- partial/failed 状态不能伪造 success；
- 重复 deletion 被拒绝；
- split 与 test 保护策略机器可读。

Stage 2 已完成极小样本数据链路。当前下一门禁是正式 train/dev/test manifest，不是直接开始完整训练。完整复现和当前状态见仓库根目录 `README.md`。
