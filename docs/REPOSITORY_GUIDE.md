# 仓库说明：每个目录和文件是做什么的

这份文档面向第一次打开仓库的人。读完后，你应该能回答四个问题：

1. 项目现在已经做到哪一步；
2. 每个目录和关键文件负责什么；
3. 想修改某个功能时应该去哪里；
4. 哪些文件可以提交 Git，哪些绝对不应该提交。

完整运行教程见 [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md)。研究原理和当前指标见仓库根目录 [`README.md`](../README.md)。按时间记录的实际开发过程见 [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md)。

## 1. 先理解当前项目是什么

最终目标是建立英文交互式发音学习平台：

```text
学习者英文语音
  -> 音素/音节识别与时间边界
  -> 与目标发音比较
  -> 映射到 VocalTractLab 发音器官轨迹
  -> 给出可解释的发音反馈
```

但是当前仓库只完成了前半段的研究基线：

- 12条 LibriSpeech 小样本的数据准备；
- MFA 离线音素对齐和确定性音节化；
- HuBERT Phone CTC 小数据过拟合；
- Guided CTC 目标音素约束对齐；
- 使用 MFA 作为边界评价参照。

当前没有完成：

- 多说话人 `dev-clean` 泛化验证；
- 可靠的音素或音节边界；
- 直接 Syllable CTC；
- Boundary/onset/nucleus/coda/stress 模型；
- 学习者发音评分；
- VTL 轨迹；
- API 和前端界面。

最重要的当前结论：12条训练音频的 Phone CTC 过拟合 PER 约为2.60%，但是与 MFA 比较的边界 MAE 约为980 ms。序列正确性门禁通过，边界门禁失败。

## 2. 两条链路不要混淆

### 2.1 MFA 数据标注链路

```text
音频 + 文本
  -> 文本规范化
  -> CMUdict/OOV G2P
  -> MFA
  -> 音素时间戳 TextGrid
  -> 确定性音节化
  -> 带时间戳音节 manifest
```

用途：离线伪标签、数据审核和边界评价参照。

### 2.2 HuBERT Phone CTC 链路

```text
音频 + 文本导出的有序音素标签
  -> HuBERT encoder
  -> Phone CTC head
  -> 音素序列
  -> Guided CTC 对齐
```

用途：训练可部署的声学模型。Phone CTC 的 loss 不读取 MFA 时间戳。

HuBERT 的原始预训练是自监督学习；本仓库用音频和文本音素标签训练 Phone CTC head，这一步属于有监督训练。

## 3. 顶层目录总览

```text
audio_process/
|-- .gitattributes
|-- .gitignore
|-- AGENTS.md
|-- README.md
|-- pyproject.toml
|-- configs/
|-- docs/
|-- schemas/
|-- scripts/
|-- src/syllable_recognition/
|-- tests/
|-- assets/       # 本地模型，Git忽略
|-- data/         # 本地数据集，Git忽略
|-- artifacts/    # 数据处理和评价产物，Git忽略
`-- runs/         # 训练checkpoint和指标，Git忽略
```

可以把它理解为：

| 目录 | 谁写入 | 保存什么 | 是否进普通Git |
| --- | --- | --- | --- |
| `configs/` | 开发者 | 可复现配置 | 是 |
| `docs/` | 开发者 | 说明、教程、日志 | 是 |
| `schemas/` | 开发者 | 公共输出格式 | 是 |
| `scripts/` | 开发者 | 下载和外部工具wrapper | 是 |
| `src/` | 开发者 | Python业务代码 | 是 |
| `tests/` | 开发者 | 单元测试和小fixture | 是 |
| `assets/` | 下载脚本 | HuBERT等大模型 | 否 |
| `data/` | 下载脚本 | LibriSpeech等音频 | 否 |
| `artifacts/` | 数据命令 | manifest、TextGrid、报告 | 否 |
| `runs/` | 训练命令 | checkpoint、metrics | 否 |

## 4. 根目录文件

### `.gitattributes`

固定跨平台换行：Python、YAML、JSON、Markdown使用LF，PowerShell脚本使用CRLF；同时把音频和模型权重声明为二进制。这样Windows checkout不会因为换行不同破坏配置SHA-256。

### `.gitignore`

忽略虚拟环境、Python缓存、编辑器配置、数据集、模型、checkpoint和实验产物。不要为了提交大文件而删除这些规则。

### `AGENTS.md`

项目的最高级工程约束。规定项目终点、Guided/Open区别、数据划分、模型路线、指标、阶段门禁和强制开发顺序。开始新的研究任务前先读它。

### `README.md`

项目总说明：研究目标、原理链路、当前结果、限制和简版复现方法。它不是逐文件教程，因此新手应该配合本文件和复现文档阅读。

### `pyproject.toml`

Python包定义和依赖入口。它完成三件事：

- 声明包名 `syllable-recognition`；
- 声明基础依赖和 `download/test/review/model-check/train` 可选依赖；
- 注册命令 `sylrec = syllable_recognition.cli:main`。

修改Python依赖或CLI入口时需要改这里。

## 5. `configs/`：所有阶段的可复现配置

配置文件不是普通备注。训练和数据命令会读取配置、校验阶段字段，并把配置哈希写入报告或checkpoint。

### `configs/contracts/guided_mvp_v1.yaml`

固定Guided MVP契约：16 kHz单声道、目标文本必填、输出schema、音节字段、数据划分和当前允许的下游命令。

### `configs/evaluation/guided_mvp_v1.yaml`

固定评价协议：

- `train-clean-100`用于训练；
- `dev-clean`用于checkpoint和阈值选择；
- `test-clean`只用于最终报告；
- Phone CTC使用PER；
- 边界使用20 ms和50 ms容差。

### `configs/data/syllabifier_arpabet_v1.yaml`

确定性ARPAbet音节化规则。定义元音核、合法onset和词内音节边界策略。英语音节划分存在歧义，所以规则必须版本化。

### `configs/data/librispeech_stage2.yaml`

MFA数据链路总配置。包含：

- 输入小样本manifest；
- 文本规范化版本；
- CMUdict/G2P策略；
- 音节化配置；
- MFA版本、模型URL、大小和SHA-256；
- corpus、TextGrid、manifest和报告输出路径；
- 12条样本和10条人工审核门禁。

### `configs/data/librispeech_phone_ctc_tiny.yaml`

无MFA时间戳的Phone CTC标签配置。它从12条文本生成有序ARPAbet序列，固定 `<blank>=0`、`<unk>=1`、padding `-100`，并只从训练集构建55类词表。

### `configs/train/hubert_phone_ctc_tiny_smoke.yaml`

一条音频、一步训练的烟雾测试。只验证前向、CTC loss、反向传播和checkpoint，不要求过拟合。

### `configs/train/hubert_phone_ctc_tiny_overfit.yaml`

一条音频、600步训练。冻结HuBERT encoder，只训练Phone CTC head，用来证明极小数据可以过拟合。

### `configs/train/hubert_phone_ctc_12_overfit.yaml`

当前12条完整正确性实验：CPU、冻结encoder、缓存hidden states、2400步、head学习率0.003。要求训练PER不超过5%、最终loss不超过1并成功恢复checkpoint。

这些配置都不是正式多说话人训练配置。仓库目前还没有1至10小时实验和完整 `train-clean-100` 配置。

## 6. `docs/`：文档所有权

### `docs/REPOSITORY_GUIDE.md`

就是当前文件，解释仓库结构和文件职责。

### `docs/REPRODUCTION_GUIDE.md`

从克隆仓库开始，一条命令一条命令复现当前结果。

### `docs/DEVELOPMENT_LOG.md`

按实际执行顺序记录每一步做了什么、遇到什么问题、产物在哪里、门禁是否通过。历史结果会保留，所以旧步骤中的数值可能和后来重跑数值略有差异。

### `docs/GIT_WORKFLOW.md`

分支命名、提交前缀、rebase/force-with-lease规则、大文件规则和bundle备份流程。

### `docs/contracts/guided_mvp_v1.md`

Guided接口契约的人类可读版本，解释请求、响应、时间戳、置信度和对齐事件语义。

## 7. `schemas/`：机器可读输出格式

### `schemas/guided_syllable_result_v1.schema.json`

Guided音节结果的JSON Schema。约束字段类型、必填字段、时间戳、phones、onset/nucleus/coda/stress、置信度和事件结构。

Schema只检查结构；跨字段语义还要由 `contracts/guided.py` 检查。

## 8. `scripts/`：外部资产和工具wrapper

### `scripts/download_hubert_model.py`

下载固定revision的 `facebook/hubert-base-ls960`，只允许三个权威文件，并校验每个文件的大小和SHA-256。

```powershell
python scripts/download_hubert_model.py
```

### `scripts/prepare_stage0_sample.py`

从固定revision的 `openslr/librispeech_asr` Parquet分片取发布顺序前12条，保存FLAC、manifest和来源报告。

```powershell
python scripts/prepare_stage0_sample.py --count 12
```

### `scripts/verify_hubert_model.py`

读取第一条真实音频的前一秒，离线运行本地HuBERT encoder，检查输入shape、hidden state shape和参数量。它不训练模型。

```powershell
python scripts/verify_hubert_model.py
```

### `scripts/download_mfa_model.ps1`

读取MFA配置中的URL、大小和SHA-256，使用HTTP Range并行下载 `english_us_arpa v3.0.0`。支持分片续传，最终必须通过哈希校验。

### `scripts/run_mfa_stage2.ps1`

Windows到WSL2的wrapper。它先运行 `mfa validate`，成功后才运行 `mfa align`，并把Windows路径转换为 `/mnt/<drive>/...`。

### `scripts/render_stage2_review.py`

把音频波形、word边界、syllable边界和ARPAbet标签画成审核图。它只生成证据，不能代替人实际查看。

## 9. `src/syllable_recognition/cli/`：命令行入口

CLI文件只做参数声明、配置路径校验、日志和调用业务函数，核心算法不写在Click命令里。

| 文件 | 负责的命令 |
| --- | --- |
| `cli/__init__.py` | 顶层 `sylrec`，注册四个命令组 |
| `cli/common.py` | JSON输出和结构化阶段日志 |
| `cli/data.py` | `sylrec data ...` |
| `cli/train.py` | `sylrec train phone-ctc` |
| `cli/align.py` | `sylrec align phone-ctc` |
| `cli/evaluate.py` | `sylrec evaluate phone-ctc-mfa` |

命令总览：

```text
sylrec data prepare-mfa
sylrec data mfa-spec
sylrec data mfa-download-spec
sylrec data build
sylrec data record-review
sylrec data validate
sylrec data build-phone-sequences
sylrec train phone-ctc
sylrec align phone-ctc
sylrec evaluate phone-ctc-mfa
```

## 10. `src/syllable_recognition/core/`：公共基础工具

### `core/artifacts.py`

统一实现：

- 文件和字节SHA-256；
- JSON读取和确定性写入；
- JSONL读取和确定性写入。

数据、训练、推理和评价都使用这个实现，避免相同产物在不同模块中出现不同序列化规则。

## 11. `src/syllable_recognition/contracts/`：公共接口契约

### `contracts/guided.py`

加载Guided配置和JSON Schema，并执行Schema无法表达的语义校验，例如：

- segment时间戳必须单调且不重叠；
- onset+nucleus+coda必须能重建phones；
- partial/failed状态必须有对应事件；
- 置信度必须在合法范围内。

## 12. `src/syllable_recognition/data/`：数据业务逻辑

### `data/normalization.py`

LibriSpeech英文文本规范化。处理大小写、标点、空白、连字符和内部撇号；没有配置数字展开策略时拒绝数字。

### `data/pronunciation.py`

把单词转换为ARPAbet：先查CMUdict，按配置选择发音变体；所有格优先复用词根；OOV显式调用 `g2p-en`，不能静默丢弃。

### `data/syllabification.py`

把一个词的ARPAbet序列按元音核和合法onset规则拆成音节，输出onset、nucleus、coda和stress。默认不跨单词重音节化。

### `data/alignment.py`

解析MFA TextGrid的word/phone tier，过滤静音，检查音素序列，并把word、phone和syllable时间戳绑定到prepared row。

### `data/stage2.py`

MFA数据链路的主要业务模块：准备corpus和词典、解析TextGrid、构建带时间戳manifest、记录人工审核、执行最终数据门禁。

### `data/phone_sequences.py`

Phone CTC数据链路：从原始文本生成有序phone标签、构建train-only词表、检查HuBERT卷积输出长度是否满足CTC路径要求，并生成报告。

### `data/ctc_dataset.py`

训练时读取FLAC和phone manifest。`PhoneCTCCollator` 动态padding波形、生成attention mask，并用 `-100` padding可变长label。

## 13. 模型、解码、训练、推理和评价

### `models/phone_ctc.py`

定义 `HubertPhoneCTC`：

```text
HuBERT encoder
  -> dropout
  -> Linear(vocabulary_size)
  -> CTC loss
```

提供冻结全部encoder和只解冻顶部若干层的接口，并统一计算卷积后的有效帧长度。

### `decoding/phone_ctc.py`

实现CTC greedy collapse、Phone Error Rate输入解码、目标音素约束Viterbi对齐，以及唯一的帧区间到秒转换。

### `metrics/per.py`

使用Levenshtein编辑距离计算语料级PER。

### `metrics/alignment.py`

比较预测phone start/end与MFA参考边界，报告MAE、最大误差和20/50 ms容差下的Precision/Recall/F1。

### `training/phone_ctc.py`

组装dataset、collator、HuBERT模型、optimizer、scheduler和训练循环。负责：

- 配置和权重哈希检查；
- MFA监督禁用检查；
- 冻结或解冻encoder；
- frozen feature缓存；
- AdamW、梯度裁剪和CTC训练；
- best/last checkpoint；
- optimizer、scheduler和随机状态保存；
- 最终loss/PER和过拟合门禁。

### `inference/phone_ctc.py`

Guided推理：目标文本经过CMUdict/G2P得到目标phones，音频经过HuBERT得到CTC logits，再做目标约束对齐，输出每个phone的emission span和置信度。

### `evaluation/phone_ctc_mfa.py`

调用Guided CTC推理并与指定MFA manifest条目比较。MFA在这里是只读评价参照，不进入训练。

## 14. `tests/`：每类测试保护什么

| 目录/文件 | 保护内容 |
| --- | --- |
| `tests/cli/test_cli.py` | 顶层命令组和子命令没有在重构中丢失 |
| `tests/contracts/test_guided_contract.py` | JSON Schema和Guided跨字段语义 |
| `tests/core/test_artifacts.py` | SHA-256、JSON、JSONL确定性 |
| `tests/data/test_normalization.py` | 文本规范化 |
| `tests/data/test_pronunciation.py` | CMUdict、所有格和OOV G2P |
| `tests/data/test_syllabification.py` | 音节核和onset规则 |
| `tests/data/test_alignment.py` | TextGrid解析和时间戳绑定 |
| `tests/data/test_stage2_review.py` | pending/fail不能错误打开人工门禁 |
| `tests/data/test_phone_sequences.py` | CTC词表、重复label路径和卷积长度 |
| `tests/models/test_phone_ctc.py` | HuBERT输出长度计算 |
| `tests/decoding/test_phone_ctc.py` | greedy collapse、PER、forced alignment和帧转秒 |
| `tests/metrics/test_alignment.py` | 边界匹配和容差指标 |
| `tests/training/test_phone_ctc.py` | checkpoint只保存可训练状态并可恢复 |
| `tests/fixtures/` | 小型固定JSON和TextGrid输入 |

当前全量测试是53项：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## 15. 本地生成目录里会出现什么

### `assets/models/facebook-hubert-base-ls960/`

本地HuBERT配置、预处理器、377 MB PyTorch权重和来源报告。

### `data/samples/librispeech_train_clean_100/`

12条FLAC、原始manifest和固定数据源报告。

### `artifacts/manifests/`

- `stage2_prepared_train.jsonl`：有phones和音节但没有MFA时间戳；
- `stage2_aligned_train.jsonl`：带MFA phone/word/syllable时间戳；
- `phone_ctc_tiny_train.jsonl`：不含MFA时间监督的Phone CTC标签。

### `artifacts/mfa/`

MFA声学模型、corpus、词典、TextGrid和alignment分析。

### `artifacts/reports/`

数据构建报告、OOV报告、人工审核队列/图片、Phone CTC数据报告和CTC对MFA评价。

### `artifacts/vocabs/`

训练集构建的55类Phone CTC词表。

### `runs/hubert-phone-ctc-*/`

每次正确性训练的解析配置、`metrics.json`、`best/checkpoint.pt` 和 `last/checkpoint.pt`。

## 16. 想改某项功能时去哪里

| 你想做什么 | 首先查看 |
| --- | --- |
| 改文本清洗 | `data/normalization.py`及对应测试 |
| 改发音词典/G2P | `data/pronunciation.py`及配置 |
| 改音节规则 | `data/syllabification.py`和syllabifier YAML |
| 改MFA数据流程 | `data/stage2.py`、`data/alignment.py`和MFA配置 |
| 改Phone词表/manifest | `data/phone_sequences.py` |
| 改HuBERT模型头 | `models/phone_ctc.py` |
| 改训练策略 | 先改训练YAML，再改 `training/phone_ctc.py` |
| 改CTC对齐 | `decoding/phone_ctc.py` |
| 改边界指标 | `metrics/alignment.py` |
| 新增CLI命令 | 对应 `cli/*.py`，核心逻辑放业务模块 |
| 改公共返回格式 | schema、contract配置和 `contracts/guided.py` 一起改 |

## 17. 绝对不要做的事情

- 不要把模型、FLAC、MFA输出或checkpoint强行 `git add -f`；
- 不要使用test-clean选择checkpoint或阈值；
- 不要从dev/test扩充训练词表；
- 不要把MFA说成在线推理模型；
- 不要把CTC emission span说成精确phone duration；
- 不要把12条训练集过拟合说成模型具有泛化能力；
- 不要在边界门禁失败时跳到VTL、API或界面制造完整演示；
- 不要在没有查看审核图或听音频时执行人工review pass。
