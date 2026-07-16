# AGENTS.md

## 1. 项目终点

本项目要构建一个英文交互式发音学习平台。系统接收学习者的连续英文语音，输出带时间戳的音节识别结果，将目标音节与 VocalTractLab（VTL）的发音器官运动信息关联，并提供接近实时、可解释的发音反馈与可视化指导。

完整业务链路：

```text
英文语音
  -> 音节边界检测与音节识别
  -> 学习者发音与目标发音比较
  -> 音节到 VTL 发音器官参数/轨迹的映射
  -> 可解释反馈与可视化
```

首要研究任务是使用 Wav2Vec 2.0、HuBERT 等开源自监督语音模型，在 LibriSpeech 上建立可复现的英文音节级识别基线。

### 1.1 现有仓库方法与取舍

递归审计表明，现有模型采用以下方法：

```text
已提前切分的单音节音频
  -> 固定长度 MFCC/Log-Mel
  -> BiLSTM 或 CNN+LSTM
  -> 整段音节分类
```

早期模型使用一个 sigmoid 向量表示 CV/CCV/CVC；后期 LibriTTS `model_2/model_2a` 将音节分成 onset、vowel、offset 三个互斥 Softmax 头。新项目只继承后期模型的组合式标签思想，不继承旧声学编码器和序列化方案。

应继承：

- 将音节拆成 onset、nucleus、coda、stress，分别预测后再组合；
- 为无 onset、无 coda 设置显式空类别；
- 对互斥类别使用独立 Softmax 和交叉熵；
- 将采样率、预处理、词表、标签分组和解码配置与模型共同发布；
- 保留小规模参考输入输出作为回归测试。

旧资产到新系统的迁移落点：

| 旧资产                                                            | 可借鉴内容                                         | 新系统落点                                                             |
| ----------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| `evoclearn/rec/vec2sym.py`                                      | 按 onset、nucleus、coda 分组编码和独立解码         | 四个版本化词表与 onset/nucleus/coda/stress 分类头                      |
| `etc/libritts_*/onsets.json`、`vowels.json`、`offsets.json` | 类别命名、显式空 onset/coda 和历史覆盖范围         | 仅作为词表审计与初始化参考；权威词表必须只由训练集构建                 |
| `Ac2Vec` 和 `compile_settings.json`                           | 模型与采样率、预处理、标签顺序、输出头配置共同发布 | 普通 YAML/JSON 配置、独立词表、checkpoint 和 schema 版本               |
| `collect/train/compile/ac2vec` CLI 分层                         | 数据准备、训练、发布、推理职责分离                 | 配置驱动、可重复运行的 data/train/evaluate/infer 命令                  |
| `ref_data/wavs` 与 `ref_data/out`                             | 小规模固定输入和历史输出                           | 自动化 golden/兼容性测试，浮点输出使用明确容差                         |
| `.gitattributes`                                                | WAV 和旧模型使用 Git LFS                           | 小规模回归资产可继续使用 LFS；数据集、缓存和 checkpoint 不进入普通 Git |

迁移时只复用上述语义和测试资产，不直接复用历史类别 ID、归一化统计或预测分数。旧 LibriTTS 模型使用 24 kHz 和固定长度 MFCC，新 LibriSpeech 链路使用 16 kHz 波形；两个域的预处理、词表与置信度不可混用。历史 vowel 词表将 stress 混入 nucleus，且 `model_2/model_2a` 口径不一致，新实现必须把 stress 拆成独立字段和分类头。

`etc/libritts_ipsyls` 中存在 Git 符号链接，在未启用符号链接支持的 Windows checkout 中可能退化为只包含目标路径的普通文本文件。新模块不得依赖这些脚本直接执行，应使用正常 Python import、模块入口或真实 wrapper 文件。

不得继承：

- 假设输入已经切成单音节；
- 将 MFCC + 固定长度 padding 作为新系统主干；
- 使用 sigmoid + MSE 处理互斥分类；
- 使用 dill 打包函数和权重，或把权重嵌入巨型 JSON；
- 依赖缺失或不一致的隐式元数据。

现有模型没有连续语音边界检测、正式数据划分、可复现的 LibriTTS 训练链路或可信的正式指标，只能作为结构参考和兼容性回归基线。已知的旧模型元数据不一致不得复制到新实现。

### 1.2 只借鉴代码形式的完整工具链

迁移目标是借鉴旧仓库的职责分层、命令组织、产物发布和回归测试形式，不借鉴旧模型的训练内容。不得复制历史 MFCC、固定长度输入、类别 ID、归一化统计、模型权重、数据切分或预测分数作为新系统默认值。

代码形式映射：

| 旧代码形式           | 新工具链职责                                                  |
| -------------------- | ------------------------------------------------------------- |
| `collect.py`       | 数据获取、文本处理、MFA 对齐、音节化、manifest 生成和数据门禁 |
| `train.py`         | 按任务训练、验证、checkpoint 保存与恢复                       |
| `compile_model.py` | 打包模型、词表、预处理、解码配置、schema 和版本信息           |
| `ac2vec.py`        | 稳定的批量/单条推理接口和服务适配层                           |
| `vec2sym.py`       | 无副作用、可单元测试的标签编码与解码函数                      |
| `ref_data`         | 小规模固定输入、预期输出和兼容性回归测试                      |

完整工程链路：

```text
项目契约与环境固定
  -> LibriSpeech 下载、校验与官方划分
  -> 文本规范化、CMUdict/OOV G2P
  -> MFA 离线音素对齐
  -> 确定性音节化与 manifest
  -> 数据门禁
  -> Phone CTC 正确性验证与训练
  -> Guided CTC 约束对齐
  -> Syllable CTC
  -> 显式边界与音节属性分类
  -> 模型评估、打包与推理
  -> 近实时 API
  -> 发音比较与 VTL
  -> 最小交互界面
```

新系统使用一个顶层 CLI 组织阶段入口，CLI 只负责配置加载、参数校验和调用业务函数，不得把核心逻辑写在命令函数中。建议命令契约：

```bash
sylrec data build --config configs/data/librispeech.yaml
sylrec data validate --manifest artifacts/manifests/train.jsonl
sylrec train phone-ctc --config configs/train/phone_ctc.yaml
sylrec evaluate phone-ctc --checkpoint runs/<run-id>/best --split dev-clean
sylrec align guided --checkpoint runs/<run-id>/best --audio sample.wav --text "HELLO WORLD"
sylrec package --run runs/<run-id> --output releases/<release-id>
sylrec serve --model releases/<release-id>
```

每个命令必须：

- 配置驱动并保存解析后的最终配置；
- 可重复运行，正确产物默认安全跳过，覆盖必须显式指定；
- 校验输入文件、manifest、词表和 checkpoint 的版本或哈希；
- 使用结构化日志，并输出可供后续阶段读取的机器可读报告；
- 明确 `guided` 或 `open`，不得依赖未记录的环境默认值；
- 将业务逻辑实现为带类型标注的可导入函数，并为关键转换添加单元测试。

每个阶段开始实现前，必须在任务说明或配置中写明：当前阶段、输入、处理、输出、数据划分、验收指标和失败后的停止条件。当前阶段门禁未通过时，不得调用下游命令制造看似完整的演示链路。

训练入口只负责组装 dataset、collator、model、optimizer、scheduler、trainer 和 evaluator。训练过程必须严格执行第 5.7、5.8 节的极小数据过拟合、冻结 encoder、逐步解冻、1 至 10 小时实验和完整 `train-clean-100` 训练顺序；指标选择遵循第 6 节，test-clean 不得参与 checkpoint、阈值或超参数选择。

工具链产物必须按所有权分开：manifest 和数据报告进入 `artifacts/`，每次训练进入独立 `runs/<run-id>/`，可部署模型进入 `releases/<release-id>/`。这些目录默认不进入普通 Git；仓库只保存配置、schema、代码、小规模回归样本及必要的校验值。详细模块布局遵循第 10 节，开发顺序遵循第 11 节。

## 2. 先确定运行模式

项目必须区分两种模式，训练、解码和评价不得混用。

### 2.1 Guided：已知目标文本

平台提前知道用户要朗读的单词或句子。这是第一版 MVP 的默认模式：

```text
目标文本 -> CMUdict/G2P -> 目标音素与音节序列
学习者语音 + 目标序列 -> CTC 约束对齐 -> 音节时间戳
对齐片段 -> 发音比较 -> VTL 轨迹 -> 反馈
```

已知文本时，不需要先解决完整的开放词汇 ASR。运行时使用可部署的 CTC forced alignment，不默认把离线 MFA 放进低延迟服务。

### 2.2 Open：未知文本

系统不知道用户说了什么，需要完成开放式识别：

```text
学习者语音
  -> 音素/音节识别
  -> 音节边界检测
  -> 音节标签与时间戳
  -> 发音分析
```

该模式还要处理插入、删除、重复、未知词和语言模型偏差。除非需求明确要求自由发言，否则不得用 Open 模式阻塞 Guided MVP。

API、manifest、训练配置和指标报告必须标明 `guided` 或 `open`。

## 3. 固定任务与输出契约

“音节级识别”包含三个不同子任务：

1. 音节边界检测：预测每个音节的 `start` 和 `end`。
2. 音节分类：预测 ARPAbet 音素组合、onset、nucleus、coda 和 stress。
3. 音节 token 化：生成 embedding 或离散 token，供比较、聚类和发音建模使用。

任何实现都必须说明覆盖了哪些子任务。不得把单词转录、音素识别、强制对齐或已切分音节分类单独描述成完整的音节级识别。

稳定输出 schema：

```json
{
  "audio_id": "sample-001",
  "mode": "guided",
  "segments": [
    {
      "start": 0.18,
      "end": 0.41,
      "phones": ["HH", "AH0"],
      "label": "HH-AH0",
      "token_id": 183,
      "confidence": 0.93
    }
  ]
}
```

第一版允许 `token_id` 为空，但必须输出时间戳、标签和置信度。

## 4. 链路一：生产 LibriSpeech 训练数据

### 4.1 数据划分

第一阶段使用：

```text
train-clean-100 -> 训练
dev-clean       -> 验证和阈值调节
test-clean      -> 最终测试
```

LibriSpeech 提供 16 kHz 英文音频、说话人信息和句子文本，但没有人工音节时间边界。保留官方说话人划分，禁止按句子随机重分数据。链路验证完成后再扩展到 360/960 小时。

### 4.2 文本规范化与音素化

```text
LibriSpeech transcript
  -> 文本规范化
  -> CMUdict/MFA dictionary 查词
  -> OOV G2P
  -> ARPAbet 音素序列
```

- CMUdict/G2P 负责文本转音素，不产生声学时间戳。
- 文本规范化、词典版本、发音选择和 OOV 来源必须记录。
- 同一个单词存在多个发音时，选择策略必须配置化并可复现。
- OOV 不得静默丢弃。

示例：

```text
HELLO WORLD
-> HH AH0 L OW1 | W ER1 L D
```

### 4.3 MFA 强制对齐

```text
音频 + 规范化文本 + 发音词典
  -> Montreal Forced Aligner
  -> 单词与音素时间戳 TextGrid
```

MFA 的职责是离线生成训练伪标签和辅助数据质检。MFA 不是文本转音素工具，也不是默认的线上推理组件。

必须记录并过滤：

- 对齐失败和缺失 TextGrid；
- OOV 或发音不匹配；
- 时间戳越界、重叠或非单调；
- 大段未对齐区域；
- 异常短、异常长或缺少元音核的片段。

### 4.4 音素到音节

第一版使用版本化的确定性英文音节化器：

```text
带时间戳 ARPAbet 音素
  -> 检测元音音节核
  -> 按固定规则分配 onset/nucleus/coda
  -> 保留或显式移除 stress
  -> 音节标签与时间区间
```

默认规则是词内音节化，不跨单词重音节化。英语音节边界存在歧义，必须固定规则和版本，不得把隐含假设散落在模型代码中。

HuBERT/Wav2Vec 2.0 默认不负责把符号音素组合成音节。只有确定性基线完成且有充分标注数据后，才考虑概率音节化模型。

示例：

```text
HH AH0 L OW1
-> HH-AH0 | L-OW1
```

### 4.5 生成 manifest

权威数据链路：

```text
LibriSpeech FLAC + transcript
  -> 文本规范化
  -> ARPAbet
  -> MFA 音素时间戳
  -> 确定性音节化
  -> train/dev/test JSONL
```

JSONL 每行表示一条完整语句：

```json
{
  "id": "19-198-0000",
  "audio": "LibriSpeech/train-clean-100/19/198/19-198-0000.flac",
  "duration": 4.82,
  "text": "HELLO WORLD",
  "speaker_id": "19",
  "phones": ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"],
  "syllables": [
    {
      "start": 0.12,
      "end": 0.29,
      "phones": ["HH", "AH0"],
      "label": "HH-AH0"
    }
  ]
}
```

CTC 训练只需要有序音素/音节标签；显式边界训练才需要 MFA 时间戳。不要为了 CTC 把全部数据切成独立 WAV。

### 4.6 数据门禁

进入正式训练前依次完成：

1. 人工构造样本的文本规范化和音节化单元测试。
2. 人工检查 10 至 100 条 LibriSpeech 样本的音素、音节和时间戳。
3. 统计 train/dev/test 的时长、说话人数、OOV、对齐失败率、音节词表和长尾分布。
4. 建立一小批人工复核的开发/测试样本，用于估计 MFA 伪标签偏差。

数据门禁未通过时，不启动完整 `train-clean-100` 训练。

## 5. 链路二：训练音节识别模型

### 5.1 训练环境

新模型使用独立的现代环境，不复用旧 TensorFlow/Keras 入口：

```text
Linux/WSL2
Python 3.10+
PyTorch + torchaudio
Hugging Face Transformers/Datasets/Accelerate
jiwer + TextGrid parser
```

依赖必须锁定，CUDA、PyTorch 和驱动版本必须匹配。第一版使用 Base 模型和 `train-clean-100`，不要在链路验证前使用 Large 或 960 小时数据。

模型选择必须配置化。首批公平比较：

```text
facebook/hubert-base-ls960
同规模英文 Wav2Vec 2.0 Base
```

数据、任务头、训练预算和评价方式必须一致。不得根据一次实验宣称某个模型全面更优。

### 5.2 Baseline A：Phone CTC

```text
语音
  -> HuBERT/Wav2Vec 2.0 encoder
  -> dropout + linear(phone_vocab_size)
  -> CTC loss
  -> ARPAbet 音素序列
  -> 确定性音节化
```

Phone CTC 用于优先验证声学模型、数据、CTC 解码和 Guided 对齐。必须正确处理 blank/pad ID、动态 padding、attention mask、输入长度、目标长度和 padding loss mask。评价使用 PER。

### 5.3 Baseline B：Guided CTC 对齐

```text
学习者语音 -> encoder -> CTC logits
目标文本 -> CMUdict/G2P -> 目标音素/音节
CTC logits + 目标序列 -> 约束对齐 -> 时间戳与置信度
```

这是发音学习 Guided MVP 的首选推理基线。必须测试正确朗读、漏读、插入、重复和明显误读，不能假设用户严格按文本朗读。

### 5.4 Baseline C：直接 Syllable CTC

```text
语音
  -> HuBERT/Wav2Vec 2.0 encoder
  -> linear(syllable_vocab_size)
  -> CTC loss
  -> 音节 token 序列
```

音节词表只能由训练集构建，dev/test 未知音节映射为 `<unk>`。必须报告词表规模、`<unk>` 比例、低频音节覆盖率以及带/不带 stress 的结果。CTC spike 只能提供粗粒度时间，不得直接称为精确音节边界。

如果完整音节词表过于稀疏，增加 onset、nucleus、coda、stress 组合式输出实验，不得读取 dev/test 扩充词表。

### 5.5 Baseline D：显式音节边界与分类

```text
完整语句
  -> HuBERT/Wav2Vec 2.0 hidden states
  -> BiLSTM/轻量 Transformer
  -> BIO 或 boundary logits
  -> 音节时间戳
  -> 分段池化
  -> onset/nucleus/coda/stress 分类
```

- 使用 MFA 音节时间戳生成帧标签。
- “秒到模型帧”只能有一个经过测试的权威实现。
- padding 帧必须从 loss 和指标中屏蔽。
- 类别不平衡可使用加权交叉熵、Focal Loss 或边界邻域软标签，策略必须配置化并做消融。
- 解码器负责平滑、峰值检测、静音过滤、最小边界间隔、短片段合并和长音频分块合并。
- 解码阈值只能使用 dev 调节。

HuBERT 在这里学习声学帧与音节边界/属性的关系，不是简单替代符号音节化器。

### 5.6 推荐主模型

最终推荐模型共享一个 HuBERT/Wav2Vec 2.0 encoder，但在单任务验证完成前保持任务头独立：

```text
16 kHz waveform
  -> HuBERT/Wav2Vec 2.0 encoder
  -> hidden states [B, T, H]
       ├─ Phone CTC head -> ARPAbet logits
       ├─ Boundary head  -> B/I/O logits
       └─ Segment pooling
            ├─ onset head
            ├─ nucleus head
            ├─ coda head
            └─ stress head
```

第一版使用最后一层隐藏状态；建立稳定基线后，可以比较中间层或可学习 layer weighted sum。不得在没有消融实验时默认最后一层或多层融合一定更优。

分段分类训练时先使用 MFA 真值区间做 pooling，确认四个分类头的上限；验证和测试必须同时报告：

- 使用 MFA 真值区间的分类结果；
- 使用模型预测区间的端到端结果。

两者差异用于衡量边界误差向分类器传播的影响。后期可以混合真值区间和预测区间训练以降低 exposure bias，但不能在基线前引入。

### 5.7 微调阶段

微调必须按阶段进行，每一阶段保留独立 checkpoint 和指标：

#### 阶段 0：正确性验证

1. 在 10 至 50 条样本上验证音频、label ID、CTC 长度、frame mask 和时间戳转换。
2. 在极小数据上过拟合，确认 loss 可接近合理下限，解码结果与标签一致。
3. 验证 checkpoint 保存、恢复和确定性评估。

无法过拟合小数据时，不得扩大训练集或更换更大模型。

#### 阶段 1：Phone CTC 适配

```text
HuBERT encoder（冻结） -> Phone CTC head
HuBERT 顶部层（逐步解冻） + Phone CTC head
```

先冻结整个 encoder 训练 CTC 头以检查数据；随后保持卷积特征前端冻结，只解冻顶部 Transformer 层。以 dev PER 选择 checkpoint。`facebook/hubert-base-ls960` 的预训练数据包含 LibriSpeech 训练语料，实验报告必须披露这一点。

#### 阶段 2：边界头训练

```text
encoder hidden states -> Boundary head -> B/I/O
```

先冻结 encoder，仅训练边界头；再解冻顶部层进行小学习率微调。使用 MFA 音节区间生成帧标签，dev 边界 F1 作为主要选择指标。边界邻域宽度、类别权重和解码阈值必须配置化。

#### 阶段 3：音节属性头训练

```text
MFA 真值音节区间
  -> segment pooling
  -> onset/nucleus/coda/stress heads
```

先冻结 encoder 和边界头，仅训练四个分类头。每个头使用独立交叉熵；无 onset/coda 使用显式 `<empty>` 类。不得把 stress 隐式混入 nucleus 后又重复计算 stress loss。

#### 阶段 4：联合多任务微调

单任务均通过门禁后，使用共享 encoder 联合训练：

```text
L_total = lambda_ctc * L_ctc
        + lambda_boundary * L_boundary
        + lambda_onset * L_onset
        + lambda_nucleus * L_nucleus
        + lambda_coda * L_coda
        + lambda_stress * L_stress
```

所有 `lambda` 必须写入配置并记录消融结果。不得默认简单相加就能平衡 CTC、帧级 loss 和音节级 loss。若某一任务梯度主导训练，先调整采样和权重，再考虑复杂的动态权重算法。

#### 阶段 5：Guided 对齐与领域适配

使用目标音素序列约束 CTC 路径，并用边界头细化音节时间戳。LibriSpeech 只能建立标准英语基线；面向学习者口音和错误发音时，需要独立的学习者语音开发集进行阈值校准和小学习率领域适配。不得用 LibriSpeech 正确朗读性能代替发音错误检测能力。

### 5.8 优化器与训练策略

每个模型按以下门禁逐步扩大：

1. 在极小样本上过拟合，验证模型、loss、mask 和解码正确。
2. 用 1 至 10 小时数据验证收敛、指标和 checkpoint 恢复。
3. 冻结整个预训练 encoder，只训练随机初始化任务头。
4. 保持卷积前端冻结，逐步解冻顶部 2 至 4 个 Transformer block。
5. 通过 dev 指标后再训练完整 `train-clean-100`。

起始配置，不视为固定最优值：

```yaml
model_name: facebook/hubert-base-ls960
sample_rate: 16000
encoder_learning_rate: 3.0e-5
head_learning_rate: 1.0e-3
layer_weight_learning_rate: 1.0e-4
epochs: 15
warmup_ratio: 0.1
weight_decay: 0.01
gradient_clip_norm: 1.0
mixed_precision: fp16
gradient_checkpointing: true
```

batch 按音频长度分桶，通过梯度累积控制有效 batch。训练至少保存 best/last checkpoint、优化器、学习率调度器和随机状态。

参数分组至少区分任务头、可学习层权重、已解冻 Transformer 层和冻结参数。encoder 学习率起点为 `1e-5` 至 `3e-5`，任务头可从 `1e-3` 起步；具体值由 dev 指标决定。使用 warmup、梯度裁剪、mixed precision 和 gradient checkpointing 时必须记录配置。

数据增强只能在基线正确后加入。SpecAugment、噪声和速度扰动必须分别做消融；任何改变时长的增强都要同步变换边界时间戳，否则会制造错误监督。

### 5.9 训练产物

每次正式实验至少保存：

```text
config.yaml
vocab.json
train/dev/test manifest 版本或校验值
文本规范化与音节化规则版本
预训练 checkpoint 名称和版本
best/last checkpoint
训练曲线与验证指标
错误样本清单
Git commit
Python/CUDA/GPU 环境信息
```

发布模型时同时提供词表、预处理配置、解码配置和最小推理示例，不能只提供权重。

## 6. 链路三：评价与模型选择

测试集只用于最终报告。模型选择、阈值和音节规则调节只能使用 train/dev。

必须报告：

- Phone CTC：PER；
- Syllable CTC：SER、`<unk>` 比例和长尾音节表现；
- 边界模型：Precision、Recall、F1，分别使用 `+/-20 ms` 和 `+/-50 ms`；
- 分割质量：漏切率、过切率和音节数量误差；
- Guided 对齐：对齐成功率、插入/删除/重复处理结果和时间戳误差；
- 系统性能：实时率、端到端延迟、峰值显存和模型大小；
- 误差分析：说话人、性别、语速、音频长度和 stress。

包含单词识别时才报告 WER。边界指标、PER、SER、CTC 对齐和时间戳匹配实现必须有单元测试。K-means、归一化统计和词表只能在训练集拟合。

## 7. 链路四：组合推理

单任务达到验收条件后再组合，不提前做复杂联合训练。

### 7.1 Guided 推理

```text
目标文本 -> CMUdict/G2P -> 目标音素/音节
学习者语音 -> 16 kHz 单声道 -> encoder -> CTC logits
目标序列 + CTC logits -> 约束对齐
对齐片段 -> 音节时间戳、标签、embedding、置信度
```

### 7.2 Open 推理

```text
学习者语音 -> 16 kHz 单声道 -> encoder
  -> 开放式 CTC/音节识别
  -> 显式边界解码
  -> 音节标签、时间戳、embedding、置信度
```

两种模式可以共享 encoder，但必须使用各自的解码器和评价集合。长音频必须支持 attention mask、分块、重叠合并。HuBERT/Wav2Vec 2.0 是双向模型，第一版“实时”定义为近实时分块推理，不得宣称严格流式。

## 8. 链路五：发音比较与 VocalTractLab

```text
已识别/对齐音节
  -> 学习者音节声学特征
  -> 目标音节声学/发音特征
  -> 差异计算
  -> VTL 目标发音轨迹
  -> 可解释反馈
```

- VTL 负责目标音节到发音器官参数/轨迹的映射，不替代语音识别。
- 必须区分静态发音位置和动态发音轨迹。
- VTL 参数、单位、采样率、说话人模型和版本必须随数据保存。
- 不得凭空构造舌、唇、腭位置。映射必须来自 VTL 合成数据、校准规则或有来源的标注数据。
- 语音识别层与 VTL 层通过稳定 schema 连接，前端不得直接依赖 VTL 原始格式。

建议中间表示：

```json
{
  "syllable": "HH-AH0",
  "duration": 0.23,
  "vtl_speaker": "JD2",
  "trajectory_rate_hz": 100,
  "parameters": {
    "HX": [],
    "HY": [],
    "JA": [],
    "LP": [],
    "LD": []
  }
}
```

## 9. 链路六：平台交付

```text
前端录音与目标文本
  -> 推理 API
  -> 音节识别/对齐服务
  -> 发音比较服务
  -> VTL 轨迹服务
  -> 可视化反馈
```

- 请求必须指定 `guided` 或 `open`；Guided 请求必须携带目标文本或目标序列。
- API 返回时间戳、标签、置信度、目标差异和可视化数据，不返回内部张量。
- 反馈必须来自模型结果或明确规则，不生成无法验证的发音建议。
- 保存用户录音前必须明确用途，日志不得默认记录原始音频或个人信息。
- 报告端到端延迟，而不只是模型前向时间。

## 10. 仓库与工程约束

当前 `evoclearn/rec` 是旧版 MFCC + TensorFlow/Keras 的已切分音节编码器。它可作为标签映射、音节向量和推理接口参考，但不能完成连续语音音节切分。

- 默认不破坏现有 `evoclearn/rec`、预训练模型和序列化格式。
- 新 PyTorch/Hugging Face 实现放在独立模块，不混用两套训练依赖。
- 修改旧模块前添加兼容测试，并说明对 `.ac2vec.json.bz2` 的影响。
- 数据集、MFA 输出、特征缓存、checkpoint 和用户录音不得提交普通 Git；需要版本管理时使用 DVC、对象存储或 Git LFS。
- 训练、评价和推理由配置驱动并支持固定随机种子。
- 音频统一为单声道 16 kHz；重采样必须显式执行并测试。
- 时间统一使用秒；公共接口提供类型标注，张量说明 shape、dtype 和 mask。
- 数据脚本必须可重复运行，正确产物可安全跳过或显式覆盖。
- 不提交密钥、绝对本机路径、下载缓存、训练日志、大型音频和 checkpoint。
- 第三方模型使用前核查模型卡、训练语言、采样率和许可证。

建议逐步形成：

```text
configs/
src/syllable_recognition/
  data/          文本、对齐、音节化和 manifest
  models/        HuBERT/Wav2Vec 2.0 与任务头
  decoding/      CTC、Guided alignment 和边界解码
  metrics/       PER、SER、边界和对齐指标
  inference/     分块推理与稳定输出接口
src/articulation/ VTL 数据、映射和比较
src/api/          平台 API
tests/
```

遵循仓库实际演进，不为匹配建议结构做无关搬迁。

## 11. 强制开发顺序

1. 固定 Guided MVP、音节定义、输出 schema 和评价协议。
2. 完成 LibriSpeech 文本规范化、ARPAbet、MFA、音节化和人工抽查。
3. 生成通过数据门禁的 train/dev/test manifest。
4. 建立 Phone CTC，并先完成极小数据过拟合与 1 至 10 小时实验。
5. 实现目标音素/音节约束下的 Guided CTC 对齐。
6. 建立直接 Syllable CTC，分析词表长尾与 `<unk>`。
7. 建立显式音节边界和 onset/nucleus/coda/stress 分类模型。
8. 组合时间戳、标签、embedding 和置信度输出。
9. 建立学习者发音比较与 VTL 稳定接口。
10. 实现近实时 Guided API 和最小交互界面。
11. Guided 模式达标后，再评估 Open 模式、联合训练和更大数据/模型。

任何代理开始实现前，必须先说明当前位于哪一阶段、输入、处理、输出、数据划分和验收指标。当前阶段的门禁未通过时，不得跳到下游阶段用临时代码掩盖上游问题。
