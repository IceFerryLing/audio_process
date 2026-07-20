# English Syllable Recognition and Guided Pronunciation Platform

本项目构建一个英文交互式发音学习平台。系统最终接收学习者的连续英文语音，输出带时间戳的音节结果，将学习者发音与目标发音比较，再通过 VocalTractLab（VTL）提供有来源、可解释的发音器官运动指导。

逐步开发记录、文件位置、运行命令和门禁结果见 [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)。

当前项目不是一个已经训练完成的产品。MFA 离线伪标签链路已经跑通；在 `feature/hubert-phone-ctc` 分支上，还实现了只使用 LibriSpeech 音频和文本音素序列的 HuBERT Phone CTC。CTC 序列过拟合门禁已经通过，但 CTC emission 边界与 MFA 的误差仍很大，因此还不能宣称音素时间划分达到可用精度。

## 1. 当前状态

截至 2026-07-20：

| 阶段 | 状态 | 已有结果 |
| --- | --- | --- |
| 0. 模型与极小数据准备 | 已完成 | HuBERT Base 本地资产；12 条、168.625 秒 LibriSpeech FLAC |
| 1. Guided MVP 契约 | 已完成 | 音节定义、输出 schema、评价协议，12 项契约测试 |
| 2. 极小样本数据标注链路 | 已完成 | 12/12 MFA 对齐；452 词、1537 音素、619 音节 |
| 2. 数据门禁 | 已通过 | 自动检查无问题；10/10 可视化抽查通过 |
| 3. 无 MFA phone-sequence manifest | 极小样本已完成 | 12 条、1537 phones、55 类 train-only 词表 |
| 3. 正式 train/dev/test manifest | 未开始 | 当前仍只有单说话人小样本 |
| 4. HuBERT Phone CTC | 正确性门禁部分通过 | 12 条冻结 encoder 过拟合 PER 2.60% |
| 5. Guided CTC 对齐 | 算法已实现，边界未达标 | MFA 对照边界 MAE 980.1 ms |
| 6. Syllable CTC | 未开始 | 尚无训练词表和模型 |
| 7. 显式边界与属性分类 | 未开始 | 尚无 Boundary/ONC/stress 模型 |
| 8. 组合推理 | 未开始 | 尚无稳定推理实现 |
| 9. 发音比较与 VTL | 未开始 | 仅有接口方向，没有轨迹数据 |
| 10. API 与界面 | 未开始 | 上游门禁通过前不构建演示链路 |

第二阶段最终校验结果：

```json
{
  "status": "passed",
  "items": 12,
  "problems": [],
  "manual_gate_passed": true,
  "passed_manual_review_items": 10,
  "failed_manual_review_items": 0,
  "pending_manual_review_items": 0,
  "downstream_training_allowed": true
}
```

这里的 `downstream_training_allowed=true` 只说明 MFA 数据转换链路正确。Phone CTC 的序列和边界指标使用各自独立门禁。

当前分支只选择一个自监督预训练模型 `facebook/hubert-base-ls960`，没有引入 Wav2Vec2。当前本地数据也只有 `train.clean.100` 发布顺序前 12 条；MFA 资产保留为评价对照，不进入 CTC loss。

## 2. 项目最终目标

完整业务链路是：

```text
英文连续语音
  -> 音节边界检测与音节识别
  -> 学习者发音与目标发音比较
  -> 目标音节到 VTL 发音器官轨迹的映射
  -> 可解释反馈与可视化
```

“音节级识别”在本项目中包含三个不同任务，不能用其中一个冒充全部能力：

1. 音节边界检测：输出每个音节的 `start` 和 `end`。
2. 音节分类：输出 phones、onset、nucleus、coda 和 stress。
3. 音节 token 化：输出 embedding 或离散 token，供比较与发音建模使用。

第一版产品优先实现 Guided 模式，Open 模式不能阻塞 Guided MVP。

### 2.1 Guided 模式

平台提前知道用户要朗读的单词或句子：

```text
目标文本
  -> 规范化
  -> CMUdict/OOV G2P
  -> 目标音素与目标音节

学习者语音
  -> HuBERT/Wav2Vec 2.0
  -> Phone CTC logits

目标序列 + CTC logits
  -> 约束对齐
  -> 音节时间戳、置信度和偏离事件
```

Guided 模式不需要先解决完整开放词汇 ASR。运行时将使用 CTC forced alignment，不把离线 MFA 放进低延迟服务。

### 2.2 Open 模式

平台不知道用户说了什么：

```text
未知文本语音
  -> HuBERT/Wav2Vec 2.0
  -> 开放式 Phone/Syllable CTC
  -> 显式边界检测
  -> 音节标签、时间戳和置信度
```

Open 模式还需要处理插入、删除、重复、未知词和语言模型偏差，只在 Guided 模式达到门禁后开始。

## 3. 两条核心链路及其区别

项目中存在两条用途完全不同、但前后衔接的链路。

### 3.1 离线训练标签生产链路

当前已经跑通的是：

```text
LibriSpeech 音频 + 已知正确转写
  -> 文本规范化
  -> CMUdict/OOV G2P
  -> 目标 ARPAbet 音素
  -> MFA forced alignment
  -> 带时间戳的音素
  -> 确定性音节化
  -> 带时间戳的音节伪标签 manifest
```

这条链路的用途是生产监督标签。MFA 必须同时拿到音频和正确文本，它不会自由识别用户说了什么，也不是最终线上模型。

### 3.2 HuBERT/Wav2Vec 模型链路

后续要训练的是：

```text
16 kHz 波形
  -> HuBERT/Wav2Vec 2.0 encoder
  -> hidden states [B, T, H]
       |-- Phone CTC head
       |-- Boundary head
       `-- segment pooling
            |-- onset head
            |-- nucleus head
            |-- coda head
            `-- stress head
```

MFA 和确定性规则生成训练答案；HuBERT/Wav2Vec 使用这些答案学习从新音频预测音素、边界和音节属性。二者关系如下：

```text
MFA + 规则                 HuBERT/Wav2Vec
离线制作伪标签     ->      使用伪标签训练
依赖正确文本               推理时主要读取语音
不是产品推理组件           最终模型 encoder
```

当前 HuBERT 只完成权重准备和 encoder smoke test，还没有参加 Phone CTC 或 Boundary 训练。

## 4. 数据标注原理

### 4.1 文本规范化

版本：`librispeech-english-v1`。

- Unicode 使用 NFKC。
- 英文统一转大写。
- 保留单词内部 apostrophe。
- hyphen 转成词边界。
- 数字没有固定展开策略时直接拒绝。
- 空文本和只含非词汇符号的文本直接拒绝。

例如：

```text
"  Hello, world!  " -> "HELLO WORLD"
"well-known"       -> "WELL KNOWN"
```

### 4.2 发音解析

版本和策略：

| 项目 | 值 |
| --- | --- |
| 词典 | `cmudict 1.1.1` |
| 多发音 | 固定选择词典第一项 |
| 所有格 | `possessive-s-v1` |
| OOV | `g2p-en 2.1.0` |
| 静默丢弃 OOV | 禁止 |

所有格先解析词根，再按词尾音素追加 `/S/`、`/Z/` 或 `/IH0 Z/`。这可以避免把 `MARGUERITE'S` 整词交给 G2P 后破坏已知词根发音。

当前 12 条样本共有 222 个唯一词，仅以下两个词使用 G2P：

```text
LESCAUT  -> L EH1 S K AO0 T
RECEVEUR -> R AH0 S IY1 V ER0
```

唯一词口径 OOV 率为 `2 / 222 = 0.009009...`。这两个外来词仍需在正式数据阶段人工审校读音。

### 4.3 音素到音节

规则版本：`arpabet-word-internal-v1`。只在单词内部音节化，不跨单词重音节化。

处理步骤：

1. 把所有带 `0/1/2` 重音数字的 ARPAbet 元音识别为 nucleus。
2. 第一个元音前的辅音成为第一个音节的 onset。
3. 最后一个元音后的辅音成为最后一个音节的 coda。
4. 两个元音之间的辅音簇使用“最长合法 onset 后缀”规则。
5. 相邻元音分别成为两个 nucleus。
6. nucleus 保存不带数字的元音，stress 单独保存。
7. 展开全部音节后必须无损还原输入音素序列。

示例一：

```text
HELLO
-> HH AH0 L OW1
-> HH-AH0 | L-OW1
```

示例二，中间辅音簇为 `K S T R`，最长合法 onset 是 `S T R`：

```text
AE1 K S T R AH0
-> AE1-K | S-T-R-AH0
```

示例三，`NG` 不是合法词内 onset：

```text
S IH1 NG ER0
-> S-IH1-NG | ER0
```

合法音素集合、空 onset/coda 语义和 onset 列表位于 `configs/data/syllabifier_arpabet_v1.yaml`。

### 4.4 MFA 时间戳如何变成音节时间戳

MFA 输出每个 word 和 phone 的区间。符号音节化先决定每个音节包含哪些 phone，再按顺序绑定 MFA phone interval：

```text
syllable.start = 第一个 phone.start
syllable.end   = 最后一个 phone.end
```

例如：

```text
HH   0.120-0.180
AH0  0.180-0.290
```

得到：

```json
{
  "start": 0.12,
  "end": 0.29,
  "phones": ["HH", "AH0"],
  "label": "HH-AH0",
  "onset": ["HH"],
  "nucleus": "AH",
  "coda": [],
  "stress": 0
}
```

构建器严格检查 word 和 phone 序列完全一致、所有 phone 恰好分配一次、时间戳单调且不重叠、时间不越出音频、每个音节都有 nucleus。

## 5. 稳定输出契约

Guided v1 公共输出由 `schemas/guided_syllable_result_v1.schema.json` 和 Python 语义校验器共同约束：

```json
{
  "schema_version": "guided-syllable-result-v1",
  "audio_id": "sample-001",
  "mode": "guided",
  "target_text": "HELLO WORLD",
  "alignment": {
    "status": "success",
    "target_syllable_count": 3,
    "aligned_syllable_count": 3,
    "events": []
  },
  "segments": [
    {
      "target_index": 0,
      "start": 0.12,
      "end": 0.29,
      "phones": ["HH", "AH0"],
      "label": "HH-AH0",
      "onset": ["HH"],
      "nucleus": "AH",
      "coda": [],
      "stress": 0,
      "token_id": null,
      "confidence": 0.95
    }
  ]
}
```

需要特别注意：

- Guided `phones/label` 描述目标文本音节，不是开放式预测。
- `confidence` 是对齐置信度，不是发音正确率。
- 插入、删除、重复和不匹配必须进入 `alignment.events`。
- 对齐失败不能伪造成功 segment。
- v1 允许 `token_id=null`。

## 6. 数据划分与评价

固定使用 LibriSpeech 官方说话人划分：

| split | 用途 |
| --- | --- |
| `train-clean-100` | 参数、词表和统计量拟合 |
| `dev-clean` | checkpoint、阈值和规则选择 |
| `test-clean` | 最终报告，禁止调参 |

禁止按句子随机重分，禁止读取 dev/test 扩充训练音节词表或拟合归一化统计。

后续必须报告：

- Phone CTC：PER。
- Syllable CTC：SER、`<unk>` 比例和长尾表现。
- 边界：`+/-20 ms`、`+/-50 ms` Precision/Recall/F1。
- 分割：漏切率、过切率和音节数量误差。
- Guided：对齐成功率、时间戳误差、插入/删除/重复处理结果。
- 系统：实时率、端到端延迟、峰值显存和模型大小。
- 切片：说话人、性别、语速、音频长度和 stress。

`facebook/hubert-base-ls960` 的预训练语料包含 LibriSpeech，正式报告必须披露这一重叠。

## 7. 仓库结构

```text
audio_process/
|-- AGENTS.md
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- assets/
|   `-- models/facebook-hubert-base-ls960/       # 本地模型，普通 Git 忽略
|-- data/
|   `-- samples/librispeech_train_clean_100/     # 本地音频，普通 Git 忽略
|-- artifacts/
|   |-- manifests/                               # 生成的 JSONL
|   |-- mfa/                                     # MFA 模型、corpus、TextGrid
|   `-- reports/                                 # 哈希、统计、审核记录
|-- configs/
|   |-- contracts/                              # Guided MVP 契约
|   |-- data/                                   # 数据准备与词表配置
|   |-- evaluation/                             # 评价协议
|   `-- train/                                  # Phone CTC 训练配置
|-- docs/
|   |-- contracts/guided_mvp_v1.md
|   |-- GIT_WORKFLOW.md
|   `-- DEVELOPMENT_LOG.md
|-- schemas/guided_syllable_result_v1.schema.json
|-- scripts/
|   |-- download_hubert_model.py
|   |-- download_mfa_model.ps1
|   |-- prepare_stage0_sample.py
|   |-- render_stage2_review.py
|   |-- run_mfa_stage2.ps1
|   `-- verify_hubert_model.py
|-- src/syllable_recognition/
|   |-- cli/                                    # Click 入口及 data/train/align/evaluate 命令
|   |-- core/                                   # JSON/JSONL 与 SHA-256 公共产物工具
|   |-- contracts/                              # Guided 输出契约
|   |-- data/                                   # 规范化、发音、音节化、MFA、CTC dataset
|   |-- decoding/                               # CTC 解码和目标约束对齐
|   |-- evaluation/                             # Phone CTC 与 MFA 对照评价
|   |-- inference/                              # Guided Phone CTC 推理
|   |-- metrics/                                # PER 与边界指标
|   |-- models/                                 # HuBERT Phone CTC 模型
|   `-- training/                               # 配置驱动训练和 checkpoint
`-- tests/                                      # 按上述模块镜像组织的单元测试
```

数据集、下载缓存、MFA 输出、checkpoint、训练日志和用户录音不进入普通 Git。

## 8. 从零复现当前结果

以下步骤复现当前 Stage 0、Stage 1 和 Stage 2。所有 PowerShell 命令从仓库根目录运行。

### 8.1 主机要求

- Windows 10/11。
- PowerShell 7 或 Windows PowerShell 5.1。
- Python 3.10+；当前验证使用 Windows Python 3.11.9。
- WSL2 Ubuntu，用于 MFA。
- Git 和网络连接。
- Stage 2 不需要 GPU。

不要复用旧 TensorFlow/Keras 环境。当前机器全局 Python 已存在 OpenCV/NumPy、TensorFlow/Keras、ml-dtypes 和 protobuf 冲突，必须使用独立虚拟环境。

### 8.2 建立 Windows Python 环境

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[download,test,review]"
```

说明：

- `download` 提供 Hugging Face 与 Parquet 小数据下载依赖。
- `test` 提供 JSON Schema 测试。
- `review` 提供波形审核图依赖。
- HuBERT 前向验证依赖体积较大，后面单独安装。

确认 CLI：

```powershell
sylrec --help
sylrec data --help
```

### 8.3 运行测试

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前预期：

```text
Ran 34 tests
OK
```

测试覆盖 Guided schema/语义、文本规范化、CMUdict/G2P、音节化、TextGrid 解析、时间戳绑定和人工门禁状态。

### 8.4 下载并校验 HuBERT Base

```powershell
python scripts/download_hubert_model.py
```

固定资产：

| 项目 | 值 |
| --- | --- |
| 模型 | `facebook/hubert-base-ls960` |
| revision | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` |
| 权重 | `pytorch_model.bin` |
| 大小 | 377,569,754 bytes |
| SHA-256 | `062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad` |
| 采样率 | 16 kHz |

正确资产会安全返回 `status=skipped`；已有文件不完整或哈希不匹配时默认停止，覆盖必须显式使用 `--overwrite`。

### 8.5 下载 12 条 LibriSpeech 小样本

```powershell
python scripts/prepare_stage0_sample.py --count 12
```

固定来源：

| 项目 | 值 |
| --- | --- |
| 数据集 | `openslr/librispeech_asr` |
| revision | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` |
| split | `train.clean.100` |
| shard | `all/train.clean.100/0000.parquet` |
| 选择 | 官方发布顺序前 12 条 |

生成：

```text
data/samples/librispeech_train_clean_100/
|-- audio/*.flac
|-- manifest.jsonl
`-- source.json
```

现有样本与哈希一致时安全跳过。更改数量或发现不匹配产物时默认停止，覆盖必须显式使用 `--overwrite`。

### 8.6 离线验证 HuBERT encoder

安装模型验证依赖：

```powershell
python -m pip install -e ".[model-check]"
python scripts/verify_hubert_model.py
```

当前预期关键结果：

```json
{
  "status": "passed",
  "input_shape": [1, 16000],
  "last_hidden_state_shape": [1, 49, 768],
  "parameter_count": 94371712
}
```

这只验证 encoder 可用，不会输出音素或音节。

### 8.7 安装 WSL2 MFA CPU 环境

如果尚未安装 Ubuntu：

```powershell
wsl --install -d Ubuntu
```

首次安装可能需要管理员权限、重启并创建 Ubuntu 用户。随后进入 Ubuntu：

```powershell
wsl -d Ubuntu
```

在 WSL bash 中安装 micromamba 2.8.1：

```bash
mkdir -p "$HOME/.local/bin"
workdir="$(mktemp -d)"
cd "$workdir"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj bin/micromamba
install -m 0755 bin/micromamba "$HOME/.local/bin/micromamba"
"$HOME/.local/bin/micromamba" --version
```

建立独立 MFA CPU 环境。以下命令来自当前环境实际安装历史：

```bash
"$HOME/.local/bin/micromamba" create -y \
  -p "$HOME/.local/share/mfa" \
  -c conda-forge \
  python=3.11 \
  montreal-forced-aligner=3.3.7 \
  'kaldi=5.5.1172=cpu*'
```

验证：

```bash
"$HOME/.local/bin/micromamba" run \
  -p "$HOME/.local/share/mfa" mfa version
```

预期 MFA 为 `3.3.7`、Python 为 3.11、Kaldi 为 5.5.1172 CPU。不要直接执行环境目录内的 `mfa`；通过 `micromamba run -p` 启动才能同时得到 `fstcompile` 等 Kaldi 工具的 PATH。

### 8.8 准备 Stage 2 文本、发音和 MFA corpus

```powershell
sylrec data prepare-mfa --config configs/data/librispeech_stage2.yaml
```

处理内容：

```text
原始 manifest
  -> 文本规范化
  -> CMUdict/OOV G2P
  -> ARPAbet
  -> 确定性音节化
  -> MFA corpus + 词典 + untimed manifest
```

主要产物：

```text
artifacts/mfa/stage2/corpus/
artifacts/mfa/stage2/lexicon.txt
artifacts/manifests/stage2_prepared_train.jsonl
artifacts/reports/stage2_prepare.json
artifacts/reports/stage2_oov.jsonl
artifacts/reports/stage2_resolved_config.yaml
```

### 8.9 下载并校验 MFA 声学模型

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/download_mfa_model.ps1
```

固定模型：

| 项目 | 值 |
| --- | --- |
| 模型 | `english_us_arpa v3.0.0` |
| 大小 | 91,928,208 bytes |
| SHA-256 | `d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b` |

脚本使用 HTTP Range 分片下载，检查每个分片尺寸、合并后总尺寸和最终 SHA-256，并复用已经完整的分片。TLS 校验没有被关闭。

### 8.10 运行 MFA validation 和 alignment

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_mfa_stage2.ps1
```

wrapper 先运行 `mfa validate`，只有 validation 成功才执行 `mfa align`。当前预期生成 12 个 TextGrid：

```text
artifacts/mfa/stage2/aligned/374/*.TextGrid
artifacts/mfa/stage2/aligned/alignment_analysis.csv
```

完整 12 个 TextGrid 已存在时脚本安全返回 `skipped`。不完整输出不会被静默覆盖，需要显式 `-Overwrite`。

### 8.11 构建带时间戳的音节 manifest

```powershell
sylrec data build `
  --config configs/data/librispeech_stage2.yaml `
  --mfa-version 3.3.7
```

输出：

```text
artifacts/manifests/stage2_aligned_train.jsonl
artifacts/reports/stage2_items.jsonl
artifacts/reports/stage2_build.json
artifacts/reports/stage2_manual_review.jsonl
```

构建结果：

| 项目 | 值 |
| --- | --- |
| 输入 | 12 |
| 对齐成功 | 12 |
| 失败 | 0 |
| 词 | 452 |
| 音素 | 1537 |
| 音节 | 619 |
| aligned manifest SHA-256 | `fa2ad552f103b6688311b1ebcf8db796d36e650e7704f482d8644d1cc175aa64` |
| TextGrid 集合 SHA-256 | `c74348dac5f24da7c7dcb942fb007716e2900e4090aed1515fb02c8bce89d213` |

### 8.12 生成并执行人工抽查

先生成包含真实波形、word 边界、syllable 边界、word 标签和完整 ARPAbet 音节标签的审核图：

```powershell
python scripts/render_stage2_review.py `
  --manifest artifacts/manifests/stage2_aligned_train.jsonl `
  --output-dir artifacts/reports/stage2_review_visuals `
  --count 10
```

审核人必须实际查看图片，再记录 pass 或 fail。下面是命令模板，不能在没有检查证据时机械执行：

```powershell
sylrec data record-review `
  --config configs/data/librispeech_stage2.yaml `
  --all-items `
  --status pass `
  --reviewer "<reviewer>" `
  --method waveform-word-syllable-arpabet-visual `
  --evidence artifacts/reports/stage2_review_visuals/stage2_review_sheet_1.png `
  --evidence artifacts/reports/stage2_review_visuals/stage2_review_sheet_2.png `
  --notes "<实际检查了什么>"
```

只有明确的 `passed_visual_review` 才计入通过。pending、failed、非法状态、重复 ID 和未知 ID 都不能打开门禁。

当前审核是视觉检查，不是人工听审。正式数据阶段还必须播放音频并抽听 TextGrid。

### 8.13 执行最终数据门禁

```powershell
sylrec data validate --config configs/data/librispeech_stage2.yaml
```

必须同时满足：

- `status=passed`；
- `problems=[]`；
- `manual_gate_passed=true`；
- `downstream_training_allowed=true`。

`stage2_build.json` 保存的是构建完成时的自动检查状态，因此人工审核前会显示 pending/false；`data validate` 根据当前审核队列计算最终门禁状态。

### 8.14 复现无 MFA 的 HuBERT Phone CTC

当前 Phone CTC 使用自监督预训练 HuBERT encoder，但音频加文本音素序列的 CTC 微调属于有监督微调。训练标签不包含 MFA 时间戳。

生成 transcript-derived phone sequence 和 train-only 词表：

```powershell
sylrec data build-phone-sequences `
  --config configs/data/librispeech_phone_ctc_tiny.yaml
```

当前数据报告：

```text
items: 12
duration: 168.625 seconds
phones: 1537
vocabulary size: 55
CTC-infeasible items: 0
MFA timestamps used: false
```

运行单步真实训练 smoke test：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_tiny_smoke.yaml
```

运行单条语音过拟合：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_tiny_overfit.yaml
```

结果为 600 步、最终 loss `0.003804`、训练 PER `0.0`。该结果只证明 loss、mask、解码、反向传播和 checkpoint 正确，不能证明泛化。

运行全部 12 条语音的冻结 encoder 过拟合：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml
```

结果：

```text
steps: 2400
training items: 12
trainable parameters: 42,295
total parameters: 94,414,007
final train loss: 0.266674
final train PER: 0.026025
checkpoint restore: passed
```

冻结 encoder 时先缓存真实 HuBERT hidden states，再反复训练随机初始化的 CTC head。这只是计算优化，不改变监督来源。

使用目标文本执行 CTC constrained alignment：

```powershell
sylrec align phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml `
  --checkpoint runs/hubert-phone-ctc-12-overfit/best/checkpoint.pt `
  --audio data/samples/librispeech_train_clean_100/audio/374-180298-0000.flac `
  --text "<target text>"
```

输出的时间语义是 `ctc_emission_span`，帧步长为 20 ms。它不是精确 phone duration，也没有使用 MFA 时间戳。

使用 MFA 作为纯评价参照：

```powershell
sylrec evaluate phone-ctc-mfa `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml `
  --checkpoint runs/hubert-phone-ctc-12-overfit/best/checkpoint.pt `
  --phone-manifest artifacts/manifests/phone_ctc_tiny_train.jsonl `
  --mfa-manifest artifacts/manifests/stage2_aligned_train.jsonl `
  --item-id 374-180298-0000 `
  --output artifacts/reports/hubert_phone_ctc_12_overfit_vs_mfa.json
```

当前第一条语音的结果：

| 指标 | 值 |
| --- | ---: |
| paired phone boundaries | 234 |
| boundary MAE | 979.74 ms |
| maximum error | 2230 ms |
| boundary F1 at 20 ms | 1.71% |
| boundary F1 at 50 ms | 3.42% |

因此目前只通过了 Phone CTC 序列正确性/过拟合门禁，边界门禁明确未通过。不得继续宣称已经获得可靠音素边界。

## 9. 产物、哈希与可重复运行

权威本地产物：

| 产物 | 位置 |
| --- | --- |
| HuBERT | `assets/models/facebook-hubert-base-ls960/` |
| 12 条音频 | `data/samples/librispeech_train_clean_100/audio/` |
| 原始小样本 manifest | `data/samples/librispeech_train_clean_100/manifest.jsonl` |
| MFA 声学模型 | `artifacts/mfa/models/english_us_arpa-v3.0.0.zip` |
| MFA corpus/词典 | `artifacts/mfa/stage2/` |
| TextGrid | `artifacts/mfa/stage2/aligned/` |
| untimed manifest | `artifacts/manifests/stage2_prepared_train.jsonl` |
| aligned manifest | `artifacts/manifests/stage2_aligned_train.jsonl` |
| 报告与审核证据 | `artifacts/reports/` |

安全重跑规则：

- 正确产物与输入、配置、哈希一致时返回 `skipped`。
- 现有产物不匹配时默认报错。
- 覆盖必须显式使用 `--overwrite` 或 `-Overwrite`。
- 配置、prepared manifest、TextGrid 集合和 aligned manifest 之间有哈希关联。
- 大型数据和模型不提交普通 Git。

## 10. 完整开发流程

后续必须按顺序推进，当前门禁未通过时不得用下游演示掩盖问题。

### 阶段 1：Guided 契约

已完成。固定输入、音节定义、公共 schema、数据划分和评价协议。

### 阶段 2：极小数据标注链路

已完成。12 条真实样本跑通规范化、发音、MFA、音节化、manifest 和审核门禁。

### 阶段 3：正式 train/dev/test manifest

当前已经在 12 条样本上实现无 MFA phone-sequence manifest；正式任务仍需：

1. 获取 `train-clean-100`、`dev-clean`、`test-clean`。
2. 分 split 生成 transcript-derived phone sequence，不使用 MFA 边界训练。
3. 只从 train 构建正式 phone vocabulary。
4. MFA 仅为 dev/test 审核子集提供评价参照。
5. 记录每条成功、失败和过滤原因。
6. 统计时长、说话人数、OOV、失败率、词表和长尾。
7. 建立 10 至 100 条人工听审集合。

门禁未通过时不能开始完整训练。

### 阶段 4：Phone CTC

```text
音频 -> HuBERT/Wav2Vec -> Phone CTC -> ARPAbet -> 音节化
```

当前已完成 1 条和 12 条冻结 encoder 过拟合，以及 checkpoint 恢复。下一步仍必须使用多说话人的 1 至 10 小时数据验证泛化，再逐步解冻顶部 2 至 4 层；最后才是完整 `train-clean-100`。主要选择指标为 dev PER。

### 阶段 5：Guided CTC 对齐

```text
目标文本 -> 目标音素
语音 -> CTC logits
目标音素 + logits -> 约束对齐 -> 时间戳和置信度
```

CTC trellis、重复 phone、backtrace 和帧到秒转换已经实现。当前 emission 边界门禁失败，必须扩大数据并测试正确朗读、漏读、插入、重复和明显误读，之后再用显式 Boundary head 细化。

### 阶段 6：直接 Syllable CTC

训练词表只从 train 构建。dev/test 未知音节映射为 `<unk>`，报告 SER、未知比例和长尾覆盖。CTC spike 只能称为粗粒度时间。

### 阶段 7：显式边界和属性分类

```text
hidden states
  -> Boundary/BIO head
  -> segment pooling
  -> onset/nucleus/coda/stress heads
```

先用 MFA 真值区间训练和评估属性头，再报告预测区间的端到端结果。边界指标使用 dev 调节阈值。

### 阶段 8：组合推理

稳定输出时间戳、标签、embedding、置信度和对齐事件。Guided 与 Open 使用不同解码器和评价集合。

### 阶段 9：发音比较与 VTL

VTL 只负责目标发音器官参数和轨迹，不替代识别。所有舌、唇、下颌参数必须来自 VTL 合成、校准规则或有来源标注，不能凭空生成。

### 阶段 10：API 与最小界面

请求必须显式选择 `guided` 或 `open`。Guided 请求携带目标文本。API 返回稳定 schema，不返回内部张量；保存用户音频前必须明确用途。

## 11. 已知限制

1. 当前 12 条数据只有一个说话人和一个 chapter，只适合正确性验证。
2. 当前没有正式 `dev-clean`、`test-clean` manifest。
3. 视觉审核没有替代人工听审。
4. `LESCAUT`、`RECEVEUR` 的 G2P 需要人工审校。
5. Phone CTC 当前只在同一个说话人的 12 条训练语音上过拟合，没有 dev 泛化结果。
6. CTC emission span 相对 MFA 的边界 MAE 约 980 ms，边界门禁未通过。
7. 当前实现是自监督预训练 encoder 加有监督 CTC 微调，不是纯无监督 ARPAbet 发现。
8. 当前没有学习者口音或错误发音开发集。
9. LibriSpeech 正确朗读性能不能代表发音错误检测能力。
10. 正式训练环境仍需在 Linux/WSL2 锁定 CUDA、PyTorch、驱动和 GPU 信息。
11. HuBERT、MFA 和后续 VTL 的模型卡、许可证与发布限制仍需单独审计。
12. 当前分支为 `feature/hubert-phone-ctc`，本轮工作区修改尚未提交。

## 12. 工作记录摘要

### 2026-07-16

- 只读审计相邻旧 `evoc-learn-rec` 仓库。
- 决定只借鉴 onset/nucleus/coda 语义、CLI 分层、发布和回归测试形式。
- 明确不继承 MFCC、固定长度、sigmoid+MSE、旧类别 ID、权重和 dill 序列化。
- 固定 Guided MVP 契约、schema 和评价协议。
- 下载并验证 HuBERT Base。
- 下载 12 条真实 LibriSpeech 小样本。

### 2026-07-17

- 实现配置驱动的 `sylrec data` CLI。
- 实现规范化、CMUdict/G2P、所有格和确定性音节化。
- 在 WSL2 CPU 环境安装 MFA 3.3.7 与 Kaldi 5.5.1172。
- 固定并校验 `english_us_arpa v3.0.0`。
- 完成 12/12 MFA 对齐、TextGrid 解析和音节 manifest。
- 生成两张波形/词/音节/ARPAbet 审核图并检查 10 条样本。
- 修复人工 review 失败状态也可能被错误计入通过的问题。
- 34 项测试全部通过。
- 补齐 HuBERT 下载、离线验证和从零复现文档。

### 2026-07-20

- 从 `MFA` 创建 `feature/hubert-phone-ctc` 分支。
- 固定只使用 `facebook/hubert-base-ls960` 和现有 LibriSpeech 小样本。
- 生成不读取 MFA 时间戳的 phone-sequence manifest 和 55 类 train-only 词表。
- 实现动态 waveform padding、attention mask、label `-100` mask 和 CTC 长度门禁。
- 实现 HuBERT Phone CTC head、greedy decode、PER、checkpoint 和恢复检查。
- 单条语音过拟合达到 loss 0.003804、PER 0.0。
- 12 条语音覆盖重跑达到 loss 0.266674、PER 2.60%。
- 实现 CTC constrained alignment 和 MFA 对照评价。
- 明确记录边界 MAE 979.74 ms，边界门禁未通过。
- 将单文件 CLI 拆分为领域命令包，并抽取无阶段依赖的产物读写公共层。
- 全量 53 项测试通过。

## 13. 下一项任务

下一项任务是取得多说话人的 1 至 10 小时 `train-clean-100` 子集和 `dev-clean` 审核子集，生成无 MFA phone-sequence manifest。随后先冻结 HuBERT 训练 CTC head，再解冻顶部层，以 dev PER 和 MFA/人工边界指标判断是否获得可迁移声学对齐。

当前 12 条过拟合已经通过，因此不再重复扩大单说话人训练步数。边界指标未通过前，不进入 Syllable CTC、VTL 或界面开发。
