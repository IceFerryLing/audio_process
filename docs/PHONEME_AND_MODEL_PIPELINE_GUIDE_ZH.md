# 音素、音节、模型传播与评测入门说明

本文面向不熟悉语音识别、音素拆分和 CTC 的读者，解释本仓库当前在做什么、文件怎样逐阶段处理、模型怎样从波形传播到音素结果，以及仓库中的保护性设计是否过于保守。

## 1. 当前阶段与能力边界

截至 2026 年 7 月 20 日，本项目仍处于 Guided MVP 的数据链路与 Phone CTC 正确性验证阶段。

已经完成或初步完成：

- 英文文本规范化；
- CMUdict 查词和 OOV G2P；
- ARPAbet 音素序列生成；
- 确定性单词内部音节化；
- MFA 离线音素与音节时间戳生成；
- HuBERT Phone CTC 极小样本过拟合；
- 已知目标文本条件下的 CTC 强制对齐；
- PER 和 CTC/MFA 边界误差评测。

尚未完成正式的 train/dev/test 全量 manifest、可用精度的音素或音节边界、Syllable CTC、音节属性分类、发音评分、VTL 轨迹和平台 API。因此当前仓库是可复现的研究基线与数据工具链，不是已经完成的发音学习产品。

## 2. 音素和音节是什么

### 2.1 音素与 ARPAbet

音素是语言中最小的发音单位。项目使用 ARPAbet 表示英语音素：

```text
CAT   -> K AE1 T
HELLO -> HH AH0 L OW1
WORLD -> W ER1 L D
```

元音后的数字表示重音：`0` 是非重读，`1` 是主重音，`2` 是次重音。项目不直接使用普通英语字母作为声学标签，因为英语拼写和发音不是一一对应的。

### 2.2 音节结构

本项目规定每个音节恰好有一个元音核，并拆成：

```text
onset + nucleus + coda + stress
```

- `onset`：元音前面的辅音；
- `nucleus`：元音核心；
- `coda`：元音后面的辅音；
- `stress`：重音等级。

例如 `CAT -> K AE1 T`：

```text
onset   = [K]
nucleus = AE
coda    = [T]
stress  = 1
label   = K-AE1-T
```

单词 `I` 的发音是 `AY1`，因此 onset 和 coda 都为空。确定性音节化实现位于 `src/syllable_recognition/data/syllabification.py`。

### 2.3 为什么要拆分

如果系统只输出“用户说的是 WORLD”，无法说明用户具体哪里发音不准确。拆成 `W ER1 L D` 后，才能分别分析开头辅音、元音、结尾辅音、重音和持续时间，并为后续 VTL 发音器官指导提供稳定的中间表示。

## 3. 两条不同的数据链路

### 3.1 MFA 离线参考标签链路

```text
LibriSpeech FLAC + transcript
  -> 文本规范化
  -> CMUdict/OOV G2P
  -> ARPAbet 音素
  -> 确定性音节化
  -> MFA 离线强制对齐
  -> TextGrid
  -> 带音素和音节时间戳的 JSONL manifest
```

MFA 接收录音、已知文本和发音词典，输出单词与音素区间。MFA 的职责是离线生成训练伪标签和评测参考，不是最终在线服务，也不负责文本转音素。

相关实现：

- `src/syllable_recognition/data/stage2.py`：准备 MFA 输入、构建和验证时间戳 manifest；
- `src/syllable_recognition/data/alignment.py`：解析 TextGrid 并校验区间；
- `configs/data/librispeech_stage2.yaml`：Stage 2 数据配置。

### 3.2 Phone CTC 训练监督链路

Phone CTC 不需要 MFA 时间戳，只需要完整录音和正确的音素顺序：

```text
FLAC + transcript
  -> 文本规范化
  -> CMUdict/OOV G2P
  -> 音素序列
  -> train-only 音素词表
  -> Phone CTC manifest
```

manifest 保存录音路径、文本和音素序列，不把完整句子切成许多单音素或单音节 WAV。

相关实现：

- `src/syllable_recognition/data/phone_sequences.py`：生成 Phone CTC manifest 和词表；
- `src/syllable_recognition/data/ctc_dataset.py`：读取完整波形和标签；
- `configs/data/librispeech_phone_ctc_tiny.yaml`：极小数据配置。

## 4. 文本侧文件处理流程

### 4.1 文本规范化

例如：

```text
Hello, world!
-> HELLO WORLD
```

当前规则会统一 Unicode 和撇号、转换为大写、将连字符转换为空格、移除非词汇标点，并拒绝没有明确展开策略的数字。数字被拒绝是为了避免系统擅自决定 `123` 应读作 `ONE TWO THREE` 还是 `ONE HUNDRED TWENTY THREE`。

实现位于 `src/syllable_recognition/data/normalization.py`。

### 4.2 CMUdict 和 G2P

项目先查询 CMUdict：

```text
HELLO -> HH AH0 L OW1
```

如果单词不在词典中，则使用 G2P 将拼写转换为音素。OOV 不会被静默丢弃，而是记录单词、生成的音素和来源。

当前多发音单词默认选择 CMUdict 的第一个发音变体。这个策略可复现，但不代表第一个变体总与实际录音一致。

实现位于 `src/syllable_recognition/data/pronunciation.py`。

### 4.3 确定性音节化

项目只在单词内部划分音节，不跨单词重新音节化：

```text
HELLO
HH AH0 L OW1
-> HH-AH0 | L-OW1
```

两个元音之间有辅音簇时，规则尽可能把合法的辅音组合分给后一个音节的 onset，其余辅音留给前一个音节的 coda。

所有输入音素必须被完整保留。音节化后重新展开的音素序列必须与输入完全相同，否则直接报错。

实现位于 `src/syllable_recognition/data/syllabification.py`，规则配置位于 `configs/data/syllabifier_arpabet_v1.yaml`。

## 5. 模型前向传播链路

训练时的数据传播过程是：

```text
完整 16 kHz 单声道波形
  -> 动态 padding + attention mask
  -> HuBERT 卷积特征提取器
  -> HuBERT Transformer encoder
  -> 每个声学时间帧的隐藏向量
  -> Dropout
  -> Linear 音素分类头
  -> 每帧音素 logits
  -> CTC loss
  -> 反向传播
```

### 5.1 Dataset 和 Collator

`PhoneCTCDataset` 根据 manifest 读取完整音频，检查采样率是否为 16 kHz、是否为单声道，并将音素转换为词表 ID。

一个 batch 中的录音长度不同，因此 collator 会补齐波形并生成 attention mask：

```text
1 = 真实音频
0 = padding
```

标签 padding 使用 `-100`，使其不进入 CTC loss。主要张量为：

```text
input_values:   [batch, samples]
attention_mask: [batch, samples]
labels:         [batch, target_phones]
```

实现位于 `src/syllable_recognition/data/ctc_dataset.py`。

### 5.2 HuBERT Encoder

波形经过 HuBERT 后变成：

```text
hidden_states: [batch, frames, hidden_size]
```

HuBERT 前面的卷积层会降低时间分辨率，所以模型为压缩后的声学帧输出隐藏特征，而不是为每个原始采样点输出结果。

### 5.3 Phone CTC 分类头

每个时间帧的隐藏向量经过 Dropout 和 Linear：

```text
[batch, frames, hidden_size]
  -> Linear
[batch, frames, vocabulary_size]
```

每个时间帧都会得到所有音素和 CTC blank 的 logits。模型实现位于 `src/syllable_recognition/models/phone_ctc.py`。

### 5.4 CTC Loss

CTC 训练只需要目标音素顺序，不需要每个音素的人工时间戳。

目标为：

```text
HH AH0 L OW1
```

模型逐帧路径可能是：

```text
blank HH HH blank AH0 AH0 blank L blank OW1 OW1
```

删除连续重复和 blank 后得到目标序列。CTC loss 会在所有能够产生目标序列的合法路径中计算概率，不要求提前指定每个音素从第几帧开始。

### 5.5 当前训练方式

当前 12 条样本过拟合实验冻结了整个 HuBERT encoder，只训练最后的 Phone CTC 分类头。这个实验不是正式泛化评测，而是验证数据、标签、长度、loss、解码和 checkpoint 能否形成最小闭环。

训练代码位于 `src/syllable_recognition/training/phone_ctc.py`。

## 6. 两种 CTC 推理方式

### 6.1 Open 贪心解码

不知道目标文本时，可以对每帧选择最大概率 token，再删除连续重复和 blank，得到预测音素序列。实现位于 `src/syllable_recognition/decoding/phone_ctc.py` 的 `collapse_ctc_ids`。

当前仓库只有基础解码能力，尚未完成正式 Open 模式。

### 6.2 Guided 强制对齐

Guided 模式提前知道用户应该朗读什么：

```text
HELLO WORLD
-> HH AH0 L OW1 W ER1 L D
```

然后在 CTC 帧概率中寻找能够完整消费目标音素序列的最高分路径：

```text
目标音素 + CTC log probabilities
  -> Viterbi CTC forced alignment
  -> 每个目标音素的发射帧区间
  -> 帧编号转换为秒
```

相关实现：

- `src/syllable_recognition/decoding/phone_ctc.py`：Viterbi CTC 强制对齐；
- `src/syllable_recognition/inference/phone_ctc.py`：加载模型并输出时间戳。

当前时间戳语义是 `ctc_emission_span`，表示模型主要在哪些帧发射该 token，不等同于真实音素完整的声学起止边界。

## 7. 评测名词说明

### 7.1 PER

PER 是 Phone Error Rate，即音素错误率：

```text
PER = 音素编辑次数 / 参考音素总数
```

编辑包括替换、删除和插入。PER 越低越好，实现位于 `src/syllable_recognition/metrics/per.py`。

当前 12 条训练样本的训练 PER 约为 2.60%。这是训练集过拟合结果，不能解释为真实测试集准确率达到 97.40%。

### 7.2 Boundary MAE

Boundary MAE 表示预测边界和参考边界平均相差多少毫秒。项目计算所有音素 start/end 边界的绝对误差并求平均。

当前单条样本的 CTC/MFA 对照结果约为：

```text
边界 MAE：979.7 ms
最大误差：2230 ms
```

这说明模型虽然能在极小数据上学习音素序列，但 CTC emission 时间位置还不能作为可靠音素边界。

### 7.3 正负 20/50 ms F1

预测边界与参考边界误差不超过指定容忍范围时视为匹配。当前单条样本约为：

```text
正负 20 ms F1：1.71%
正负 50 ms F1：3.42%
```

当前 `paired_phone_boundary_metrics` 要求参考和预测音素序列完全一致，再按位置一一配对，所以 precision、recall 和 F1 数值相同。这是正确性阶段的简化指标，不是未来处理插入、删除和错读时的完整边界指标。

实现位于 `src/syllable_recognition/metrics/alignment.py`。

### 7.4 Confidence

当前 confidence 是模型在分配给目标音素的发射帧上，对该音素给出的平均概率。它不是发音正确率、口音评分、标准发音相似度或 VTL 器官位置准确度。

## 8. MFA 与 HuBERT 的关系

```text
MFA：录音 + 正确文本 + 发音词典 -> 离线参考时间戳
HuBERT Phone CTC：录音 -> 帧级音素概率
```

当前 Phone CTC 训练没有使用 MFA 时间戳，只使用正确的音素顺序。MFA 时间戳在训练后作为外部参考评价 CTC emission 边界。

## 9. 文件和产物流转

- `configs/`：阶段参数、输入输出路径、模型、随机种子和冻结策略；
- `data/`：原始或小规模数据样本；
- `artifacts/`：manifest、TextGrid、词表、数据报告和评测报告；
- `runs/`：每次训练的配置、指标、best 和 last checkpoint；
- `src/syllable_recognition/data/`：文本、发音、音节化、MFA 和 manifest；
- `src/syllable_recognition/models/`：HuBERT 和任务头；
- `src/syllable_recognition/training/`：训练与 checkpoint；
- `src/syllable_recognition/decoding/`：CTC 解码和强制对齐；
- `src/syllable_recognition/inference/`：单条音频推理；
- `src/syllable_recognition/metrics/`：PER 和边界指标；
- `src/syllable_recognition/contracts/`：输出 schema 和语义校验；
- `src/syllable_recognition/cli/`：命令行调度。

## 10. 这个项目是否过于保守

简短结论是：**代码确实非常保守，但大部分保护措施在当前研究阶段是有意且合理的；少数地方如果原样进入正式产品，会显得过严，需要在基线验证后逐步放宽。**

### 10.1 合理且应当保留的保护措施

#### 明确区分 Guided 和 Open

已知目标文本的强制对齐与未知文本的开放识别不是同一个问题。强制配置标明模式，可以避免把 Guided 结果误称为开放识别能力。

#### train/dev/test 隔离

训练集构建词表，开发集选择 checkpoint 和阈值，测试集只做最终报告。这是防止数据泄漏的基本要求，不属于过度设计。

#### 配置、词表和文件哈希

语音实验很容易出现“代码相同，但词表顺序、manifest 或预训练模型已经变化”的问题。保存解析后配置和 SHA-256，可以防止拿错词表或 checkpoint 后仍得到看似正常的输出。

#### 采样率、声道和时间戳检查

16 kHz 与 24 kHz 混用、双声道误读、时间戳越界或重叠都会制造隐蔽错误。这里直接停止通常比静默继续训练更安全。

#### OOV 不静默丢弃

如果无法查词的单词被直接删除，录音和标签序列会错位。显式记录 G2P 来源是必要的。

#### 音节化必须无损

音节拆分后重新展开必须等于原音素序列。否则辅音可能在规则处理中凭空消失，而后续指标仍可能看起来正常。

#### MFA 不进入 Phone CTC loss

Phone CTC 只学习音素顺序，MFA 单独作为边界参考，更容易判断模型到底学到了什么，也避免训练和评价形成循环论证。

#### 小数据过拟合门禁

如果模型连 10 至 50 条样本都无法记住，扩大到 100 小时通常只会浪费计算资源。先验证最小闭环是合理的工程习惯。

#### 安全跳过和显式覆盖

数据处理产物生成成本较高，默认不覆盖已有结果可以避免误删或混写实验产物。

### 10.2 当前可能偏保守的地方

#### 文本规范化直接拒绝数字

对干净基线来说这是安全的，但正式产品最终需要版本化的数字、金额、年份、缩写和符号展开策略，不能永久依赖报错。

#### 多发音单词固定选择第一个词典变体

这保证了确定性，但实际录音可能采用第二或第三个合法发音。长期需要结合上下文和声学分数选择发音，否则对齐失败可能只是词典变体选错。

#### MFA 标签必须与预期音素完全一致

当前构建过程要求 TextGrid 音素序列与文本侧音素序列严格相等。它适合作为干净数据门禁，但可能拒绝缩读、连读、口音变体和词典模型差异。学习者语音不能永久使用这一严格假设。

#### 边界评测要求标签完全一致

当前 paired 指标只适合正确性阶段。正式 Guided 评测需要先处理插入、删除、重复和错读，再计算可匹配部分的时间误差。

#### 仅允许本地预训练模型

禁止训练过程隐藏下载有利于复现，但初次使用成本较高。可以保留显式下载命令，同时继续禁止训练时自动联网。

#### 门禁较多，迭代速度较慢

契约、数据、人工复核、过拟合、边界和正式训练逐层设置门禁，会比快速制作演示慢。但项目最终要向学习者提供发音反馈和发音器官指导；如果上游边界错误，下游反馈可能具有误导性，因此当前阶段宁可慢也不应制造虚假的完整链路。

### 10.3 真正需要警惕的风险

真正的风险不是“检查太多”，而是把正确性阶段的严格假设永久固化成产品假设，例如：

- 永远只使用词典第一发音；
- 永远要求用户完整、准确地朗读目标文本；
- 永远要求音素标签完全一致后才计算时间误差；
- 把 CTC emission span 当成真实声学边界；
- 把 CTC confidence 当成发音正确率；
- 因为训练集 PER 很低就跳过 dev/test 泛化评测。

这些做法在极小数据基线中可以接受，但后续必须替换成能够处理真实学习者错误的算法和指标。

### 10.4 建议的调整方式

不建议一次性删除保护措施，而应把检查分成三种等级：

```text
fatal：会破坏标签、模型兼容性或导致数据泄漏，必须停止
warning：可能是合法语音变体，记录后允许继续
report：只进入统计和误差分析，不阻塞实验
```

建议保持 fatal 的情况：

- 采样率或声道错误；
- 词表与 checkpoint 哈希不匹配；
- CTC 输入长度小于目标长度；
- 时间戳越界或严重重叠；
- test 数据参与模型和阈值选择。

后续可以降为 warning 或 report 的情况：

- 多发音词典变体；
- 少量 OOV；
- MFA 与词典存在可解释的小差异；
- 学习者发生插入、删除、重复或缩读；
- 某个样本对齐失败，但整体数据仍可统计和继续处理。

## 11. 总体判断

从工程取向上，可以粗略理解为：

```text
70% 是合理的研究可复现和数据安全保护
20% 是正确性阶段的临时严格假设
10% 可能在后续造成开发摩擦，需要适时简化
```

这不是精确统计，而是对当前设计取向的判断。

如果目标是几天内制作一个演示页面，这套流程显得过于保守；如果目标是建立可复现的音节识别基线，并最终向学习者提供不能误导人的发音反馈，那么这种保守性总体上是必要的。

## 12. 一句话总结完整链路

```text
目标文本
  -> 规范化
  -> CMUdict/G2P
  -> ARPAbet 音素
  -> 确定性音节

16 kHz 波形
  -> HuBERT
  -> 帧级隐藏特征
  -> Phone CTC logits

目标音素 + CTC logits
  -> Guided 强制对齐
  -> 音素时间戳和置信度
  -> 音节组合
  -> 后续发音比较与 VTL

MFA TextGrid
  -> 离线参考时间戳
  -> 数据检查和边界评价
```

当前最重要的实验结论是：模型已经能够在极小训练集上学习音素序列，但 CTC emission 边界与 MFA 参考边界误差仍然很大，因此暂时不能宣称系统已经实现可靠的音素或音节时间切分。

## 13. ARPAbet 与 IPA 双版本接口

`src/syllable_recognition/data/transcription.py` 提供两个可导入的 Python 接口：

```python
from pathlib import Path

from syllable_recognition.data import PhoneticTranscriber


transcriber = PhoneticTranscriber.from_default_packages(
    Path("configs/data/syllabifier_arpabet_v1.yaml")
)

arpabet = transcriber.transcribe_arpabet("Hello world")
ipa = transcriber.transcribe_ipa("Hello world")
```

关键结果：

```text
ARPAbet: HH AH0 L OW1 | W ER1 L D
IPA:     həˈloʊ | ˈwɝld
```

也可以调用函数式接口 `transcribe_arpabet()` 和 `transcribe_ipa()`，并显式传入 `PronunciationResolver` 与 `Syllabifier`。

ARPAbet 接口是 manifest、CTC、MFA 和 checkpoint 使用的权威标签视图。IPA 接口是从同一 ARPAbet 发音和音节结构派生的宽式美式 IPA 显示视图，不改变训练词表，也不表示学习者实际发出的精细音值。

Windows PowerShell 需要正确的 UTF-8 输出环境才能直接打印 IPA，例如：

```powershell
$env:PYTHONIOENCODING = "utf-8"
```
