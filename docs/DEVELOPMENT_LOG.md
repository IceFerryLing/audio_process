# 开发日志

本文档按实际执行顺序记录项目每一步完成了什么、代码写在哪里、生成了什么产物、如何验证，以及当前是否允许进入下一阶段。未实现的能力不会写成已完成。

## 1. 当前结论

| 项目 | 当前值 |
| --- | --- |
| 当前日期 | 2026-07-20 |
| 当前分支 | `feature/hubert-phone-ctc` |
| 选择的模型 | `facebook/hubert-base-ls960` |
| 当前数据 | LibriSpeech `train.clean.100` 前12条 |
| 当前训练方式 | 自监督预训练 HuBERT encoder + 有监督 Phone CTC 微调 |
| MFA 用途 | 离线伪标签基线和评价参照，不进入 Phone CTC loss |
| Phone CTC 序列门禁 | 通过 |
| Phone CTC 边界门禁 | 未通过 |
| 全量测试 | 53/53 通过 |

本分支没有引入 Wav2Vec2。模型目录中只有 HuBERT Base，本地样本目录中只有当前 LibriSpeech 小样本。数据、模型、MFA 输出、运行目录和 checkpoint 均由 `.gitignore` 排除，不进入普通 Git。

## 2. 2026-07-16：项目准备

### 步骤 1：审计旧仓库

输入：桌面相邻的旧 `evoc-learn-rec` 仓库。

完成内容：

- 确认旧模型处理的是已经切成单音节的短音频。
- 确认旧声学主干是固定长度 MFCC/Log-Mel 加 BiLSTM 或 CNN+LSTM。
- 确认旧模型不能处理连续语音音节边界。
- 决定只借鉴 onset/nucleus/coda 标签语义、CLI 分层、模型发布形式和回归测试形式。
- 明确不迁移旧 MFCC、固定长度输入、类别 ID、归一化统计、权重、sigmoid+MSE 和 dill 序列化。

记录位置：

- `AGENTS.md`
- `README.md`

状态：完成，只读审计，没有修改旧仓库。

### 步骤 2：固定 Guided MVP 契约

输入：项目目标和强制开发顺序。

完成内容：

- 固定 `guided` 为第一版模式。
- 固定单声道16 kHz输入和秒级时间单位。
- 固定音节字段：phones、onset、nucleus、coda、stress。
- 固定插入、删除、重复和 mismatch 事件语义。
- 固定 `train-clean-100/dev-clean/test-clean` 用途。
- 禁止使用 test-clean 调参。

源码和配置位置：

- `configs/contracts/guided_mvp_v1.yaml`
- `configs/evaluation/guided_mvp_v1.yaml`
- `configs/data/syllabifier_arpabet_v1.yaml`
- `schemas/guided_syllable_result_v1.schema.json`
- `src/syllable_recognition/contracts/guided.py`
- `docs/contracts/guided_mvp_v1.md`

测试位置：

- `tests/contracts/test_guided_contract.py`
- `tests/fixtures/contracts/guided_valid.json`
- `tests/fixtures/contracts/guided_invalid_overlap.json`

验证结果：12项契约测试通过。

状态：Stage 1 门禁通过。

### 步骤 3：下载并固定 HuBERT Base

输入：`facebook/hubert-base-ls960`。

完成内容：

- 固定 Hugging Face revision。
- 只保存 PyTorch 权重和必要配置。
- 增加大小与 SHA-256 校验。
- 增加正确产物安全跳过和显式覆盖机制。
- 增加一秒音频离线前向验证。

脚本位置：

- `scripts/download_hubert_model.py`
- `scripts/verify_hubert_model.py`

本地产物位置：

- `assets/models/facebook-hubert-base-ls960/config.json`
- `assets/models/facebook-hubert-base-ls960/preprocessor_config.json`
- `assets/models/facebook-hubert-base-ls960/pytorch_model.bin`
- `assets/models/facebook-hubert-base-ls960/source.json`

固定信息：

```text
revision: dba3bb02fda4248b6e082697eee756de8fe8aa8a
weight size: 377,569,754 bytes
weight SHA-256: 062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad
parameters: 94,371,712
```

离线验证：

```text
input shape: [1, 16000]
last hidden state: [1, 49, 768]
```

状态：模型资产通过验证。此时只有 encoder，没有 Phone CTC head。

### 步骤 4：下载 LibriSpeech 小样本

输入：Hugging Face `openslr/librispeech_asr`。

完成内容：

- 固定数据 revision 和源 Parquet shard。
- 按官方发布顺序选择 `train.clean.100` 前12条。
- 保留原始 FLAC。
- 校验每条音频的采样率、声道、时长和 SHA-256。
- 保留官方 split，不随机重分。

脚本位置：

- `scripts/prepare_stage0_sample.py`

本地产物位置：

- `data/samples/librispeech_train_clean_100/audio/`
- `data/samples/librispeech_train_clean_100/manifest.jsonl`
- `data/samples/librispeech_train_clean_100/source.json`

数据结果：

```text
items: 12
duration: 168.625 seconds
sample rate: 16 kHz
channels: 1
speaker: 374
chapter: 180298
```

状态：Stage 0 小数据准备完成。该数据只适合正确性验证。

## 3. 2026-07-17：MFA 数据链路

### 步骤 5：实现文本规范化

输入：LibriSpeech 原始转写。

完成内容：

- NFKC Unicode 规范化。
- 英文统一大写。
- 保留内部 apostrophe。
- hyphen 拆成词边界。
- 没有固定策略的数字直接拒绝。
- 空文本和非词汇文本直接拒绝。

源码位置：

- `src/syllable_recognition/data/normalization.py`

测试位置：

- `tests/data/test_normalization.py`

状态：通过。

### 步骤 6：实现 CMUdict 和 OOV G2P

输入：规范化英文单词。

完成内容：

- 固定 `cmudict 1.1.1`。
- 多发音词固定选择第一项。
- 实现 `possessive-s-v1` 所有格策略。
- 固定 `g2p-en 2.1.0` 作为 OOV fallback。
- 禁止 OOV 静默丢弃。
- 禁止 G2P 隐式在线下载 NLTK 资源。

源码位置：

- `src/syllable_recognition/data/pronunciation.py`

测试位置：

- `tests/data/test_pronunciation.py`

当前 OOV：

```text
LESCAUT  -> L EH1 S K AO0 T
RECEVEUR -> R AH0 S IY1 V ER0
```

状态：通过。外来词读音仍需正式人工审校。

### 步骤 7：实现确定性音节化

输入：单词内部的 ARPAbet phone sequence。

完成内容：

- 每个带重音元音形成一个 nucleus。
- 词首辅音进入第一个 onset。
- 词尾辅音进入最后一个 coda。
- 元音之间使用最长合法 onset 后缀规则。
- stress 与 nucleus 分开保存。
- 音节展开后必须无损恢复输入 phone sequence。

源码和配置位置：

- `src/syllable_recognition/data/syllabification.py`
- `configs/data/syllabifier_arpabet_v1.yaml`

测试位置：

- `tests/data/test_syllabification.py`

状态：通过。

### 步骤 8：安装并固定 MFA

输入：WSL2 Ubuntu。

完成内容：

- 使用 micromamba 独立 CPU 环境。
- 固定 Python 3.11、MFA 3.3.7 和 Kaldi 5.5.1172 CPU。
- 固定 `english_us_arpa v3.0.0`。
- 实现 Range 分片下载、尺寸检查和 SHA-256 检查。
- wrapper 先执行 `mfa validate`，再执行 `mfa align`。

脚本位置：

- `scripts/download_mfa_model.ps1`
- `scripts/run_mfa_stage2.ps1`

配置位置：

- `configs/data/librispeech_stage2.yaml`

模型位置：

- `artifacts/mfa/models/english_us_arpa-v3.0.0.zip`

模型 SHA-256：

```text
d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b
```

状态：MFA 环境和模型通过验证。

### 步骤 9：生成 MFA corpus 和对齐

执行命令：

```powershell
sylrec data prepare-mfa --config configs/data/librispeech_stage2.yaml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_mfa_stage2.ps1
```

完成内容：

- 生成规范化 `.lab`。
- 生成222词自定义词典。
- 生成 untimed manifest。
- 对12条音频执行 MFA validation 和 alignment。
- 生成12个 TextGrid。

业务代码位置：

- `src/syllable_recognition/data/stage2.py`
- `src/syllable_recognition/data/alignment.py`

产物位置：

- `artifacts/mfa/stage2/corpus/`
- `artifacts/mfa/stage2/lexicon.txt`
- `artifacts/mfa/stage2/aligned/`
- `artifacts/manifests/stage2_prepared_train.jsonl`

状态：12/12 对齐成功。

### 步骤 10：构建音节时间戳 manifest

执行命令：

```powershell
sylrec data build `
  --config configs/data/librispeech_stage2.yaml `
  --mfa-version 3.3.7
```

完成内容：

- 严格匹配 word 和 phone sequence。
- 把 MFA phone interval 绑定到确定性音节。
- 检查时间戳单调、不重叠、不越界。
- 检查每个音节都有 nucleus。
- 检查音节展开后无损恢复全部 phones。

产物位置：

- `artifacts/manifests/stage2_aligned_train.jsonl`
- `artifacts/reports/stage2_items.jsonl`
- `artifacts/reports/stage2_build.json`
- `artifacts/reports/stage2_manual_review.jsonl`

结果：

```text
items: 12/12
words: 452
phones: 1537
syllables: 619
failed items: 0
```

状态：自动门禁通过。

### 步骤 11：执行 MFA 人工视觉审核

完成内容：

- 绘制真实波形。
- 叠加 word 边界和标签。
- 叠加 syllable 边界和 ARPAbet 标签。
- 检查前10条样本。
- 修复“failed review 也可能被计入通过”的门禁漏洞。

脚本和代码位置：

- `scripts/render_stage2_review.py`
- `src/syllable_recognition/data/stage2.py`
- `tests/data/test_stage2_review.py`

审核证据位置：

- `artifacts/reports/stage2_review_visuals/stage2_review_sheet_1.png`
- `artifacts/reports/stage2_review_visuals/stage2_review_sheet_2.png`

最终状态：10/10 `passed_visual_review`。这不是人工听审。

## 4. 2026-07-20：HuBERT Phone CTC 分支

### 步骤 12：创建独立开发分支

完成内容：

- 从 `MFA` 分支创建 `feature/hubert-phone-ctc`。
- 只选择 `facebook/hubert-base-ls960`。
- 不引入 Wav2Vec2。
- 保留 MFA 作为评价基线。

状态：完成。当前修改尚未提交。

### 步骤 13：生成无 MFA Phone CTC 数据

输入：原始 LibriSpeech 音频和文本，不读取 TextGrid。

处理链路：

```text
audio + transcript
-> normalize transcript
-> CMUdict/OOV G2P
-> ordered ARPAbet sequence
-> train-only vocabulary
```

完成内容：

- 新增 `sylrec data build-phone-sequences`。
- 生成不含 MFA 时间戳的 sequence manifest。
- 构建 train-only phone vocabulary。
- 固定 `<blank>=0`、`<unk>=1`、label padding `-100`。
- 保留 vowel stress。
- 根据 HuBERT convolution stride 检查 CTC 最小路径长度。
- 报告中显式记录 `mfa_timestamps_used=false`。

源码和配置位置：

- `src/syllable_recognition/data/phone_sequences.py`
- `configs/data/librispeech_phone_ctc_tiny.yaml`
- `src/syllable_recognition/cli/data.py`

测试位置：

- `tests/data/test_phone_sequences.py`

产物位置：

- `artifacts/manifests/phone_ctc_tiny_train.jsonl`
- `artifacts/vocabs/phone_ctc_tiny.json`
- `artifacts/reports/phone_ctc_tiny_data.json`
- `artifacts/reports/phone_ctc_tiny_data_config.yaml`

结果：

```text
items: 12
duration: 168.625 seconds
phones: 1537
vocabulary size: 55
CTC-infeasible items: 0
```

状态：数据正确性门禁通过。当前不是正式 train/dev/test 数据。

### 步骤 14：实现动态 CTC dataset 和 collator

完成内容：

- 从 JSONL 动态读取 FLAC。
- 使用 HuBERT feature extractor 归一化和 padding。
- 生成 waveform attention mask。
- 使用 `-100` 屏蔽 label padding。
- 保存 reference phones 供 PER 计算。

源码位置：

- `src/syllable_recognition/data/ctc_dataset.py`

真实两条音频 collator 验证：

```text
input_values: [2, 257360]
attention_mask: [2, 257360]
labels: [2, 140]
```

状态：通过。

### 步骤 15：实现 HuBERT Phone CTC

完成内容：

- 本地加载固定 HuBERT encoder。
- 增加 dropout 和线性 phone head。
- 增加 CTC loss。
- 正确计算 encoder output length。
- 支持冻结整个 encoder。
- 支持只解冻顶部 Transformer 层。
- 冻结 encoder 时使用 `no_grad`。
- 支持缓存冻结 encoder hidden states。

源码位置：

- `src/syllable_recognition/models/phone_ctc.py`

测试位置：

- `tests/models/test_phone_ctc.py`

模型规模：

```text
total parameters: 94,414,007
trainable head parameters: 42,295
```

状态：模型前向、loss 和长度转换通过。

### 步骤 16：实现解码和 PER

完成内容：

- greedy argmax。
- CTC duplicate collapse。
- blank 删除。
- 重复 phone 被 blank 分隔时正确保留。
- token-level Levenshtein distance。
- corpus-level PER。

源码位置：

- `src/syllable_recognition/decoding/phone_ctc.py`
- `src/syllable_recognition/metrics/per.py`

测试位置：

- `tests/decoding/test_phone_ctc.py`

状态：通过。

### 步骤 17：实现配置驱动训练

完成内容：

- 新增 `sylrec train phone-ctc`。
- 固定随机种子。
- 区分 head 和 encoder 学习率。
- 使用 AdamW 和 constant scheduler。
- 使用梯度裁剪。
- 保存 best/last checkpoint。
- 保存 optimizer、scheduler、Torch/Python 随机状态。
- checkpoint 只保存实际可训练参数，encoder 由固定本地权重恢复。
- 校验 config、manifest、vocabulary 和 model weight 哈希。
- 训练时再次拒绝任何 MFA timestamp supervision。
- 配置与产物一致时安全跳过。

源码位置：

- `src/syllable_recognition/training/phone_ctc.py`
- `src/syllable_recognition/cli/train.py`

配置位置：

- `configs/train/hubert_phone_ctc_tiny_smoke.yaml`
- `configs/train/hubert_phone_ctc_tiny_overfit.yaml`
- `configs/train/hubert_phone_ctc_12_overfit.yaml`

测试位置：

- `tests/training/test_phone_ctc.py`

运行产物位置：

- `runs/hubert-phone-ctc-tiny-smoke/`
- `runs/hubert-phone-ctc-tiny-overfit/`
- `runs/hubert-phone-ctc-12-overfit/`

状态：训练、checkpoint 和恢复链路通过。

### 步骤 18：执行单条语音过拟合

执行命令：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_tiny_overfit.yaml
```

结果：

```text
training items: 1
steps: 600
final loss: 0.003804
final PER: 0.0
checkpoint restore: passed
```

状态：单条正确性门禁通过。该结果不代表泛化。

### 步骤 19：执行12条语音过拟合

执行命令：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml
```

结果：

```text
training items: 12
steps: 2400
final loss: 0.266774
final PER: 0.026025
checkpoint restore: passed
```

指标位置：

- `runs/hubert-phone-ctc-12-overfit/metrics.json`

状态：12条序列过拟合门禁通过。数据仍只有一个说话人。

### 步骤 20：实现 CTC 目标约束对齐

完成内容：

- 构建 blank 交错的 CTC target state sequence。
- 实现 Viterbi trellis。
- 实现 backtrace。
- 支持相邻重复 phone。
- 输出每个目标 phone 的 emission frame span 和置信度。
- 唯一帧到秒转换使用 HuBERT 320 sample hop，即20 ms。
- 输出显式标记 `timestamp_semantics=ctc_emission_span`。

源码位置：

- `src/syllable_recognition/decoding/phone_ctc.py`
- `src/syllable_recognition/inference/phone_ctc.py`
- `src/syllable_recognition/cli/align.py`

CLI：

```text
sylrec align phone-ctc
```

测试位置：

- `tests/decoding/test_phone_ctc.py`

状态：算法和端到端命令通过。emission span 不是精确 phone duration。

### 步骤 21：使用 MFA 做边界对照

完成内容：

- MFA 只作为 evaluation reference。
- 检查 CTC 和 MFA phone 序列一致。
- 计算 start/end 配对边界误差。
- 报告 boundary MAE、最大误差、20 ms F1 和50 ms F1。
- 报告显式记录 `mfa_timestamps_used_for_training=false`。

源码位置：

- `src/syllable_recognition/evaluation/phone_ctc_mfa.py`
- `src/syllable_recognition/metrics/alignment.py`
- `src/syllable_recognition/cli/evaluate.py`

测试位置：

- `tests/metrics/test_alignment.py`

评价产物位置：

- `artifacts/reports/hubert_phone_ctc_12_overfit_vs_mfa.json`

第一条语音结果：

```text
phone count: 117
paired boundaries: 234
boundary MAE: 980.09 ms
maximum error: 2230 ms
20 ms F1: 1.71%
50 ms F1: 3.42%
```

状态：序列门禁通过，边界门禁未通过。不能宣称已经得到可靠音素边界。

### 步骤 22：完成回归验证和文档

完成内容：

- 运行全部单元测试。
- 编译全部 Python 文件。
- 解析全部 JSON、YAML 和 TOML。
- 验证数据构建安全跳过。
- 验证训练运行安全跳过。
- 验证 checkpoint 包含 optimizer、scheduler 和随机状态。
- 更新 README 的原理、复现命令、结果和停止条件。

文档位置：

- `README.md`
- `docs/DEVELOPMENT_LOG.md`
- `docs/contracts/guided_mvp_v1.md`

最终结果：

```text
tests: 48/48 passed
compile: passed
config parsing: passed
data safe rerun: passed
training safe rerun: passed
```

状态：当前分支工程实现完成，边界质量门禁保持失败状态。

### 步骤 23：重构项目结构并保持命令兼容

当前阶段：工程结构重构，不改变 Phone CTC 研究阶段和门禁结论。

输入：现有 `src/syllable_recognition` 业务模块、单文件 CLI、48 项测试，以及已生成的数据和训练产物。

处理：

- 将 `src/syllable_recognition/cli.py` 拆为 `cli/` 命令包。
- `cli/__init__.py` 只组装顶层命令和日志初始化。
- `cli/data.py`、`train.py`、`align.py`、`evaluate.py` 分别拥有各自命令。
- `cli/common.py` 统一机器可读 JSON 输出和结构化阶段日志。
- 新增 `core/artifacts.py`，统一 JSON、JSONL 和 SHA-256 读写。
- 数据、训练、推理和评价模块改用公共产物工具。
- 保留 `sylrec = "syllable_recognition.cli:main"`，所有命令名称和参数不变。
- 保留配置路径、manifest、报告、checkpoint 和运行目录不变。

源码位置：

- `src/syllable_recognition/cli/`
- `src/syllable_recognition/core/artifacts.py`
- `src/syllable_recognition/data/stage2.py`
- `src/syllable_recognition/data/phone_sequences.py`
- `src/syllable_recognition/data/ctc_dataset.py`
- `src/syllable_recognition/training/phone_ctc.py`
- `src/syllable_recognition/inference/phone_ctc.py`

测试位置：

- `tests/cli/test_cli.py`
- `tests/core/test_artifacts.py`

输出与验证：

```text
editable install: passed
sylrec/data/train/align/evaluate help: passed
tests: 53/53 passed
compile: passed
data safe rerun: skipped
training safe rerun: skipped
```

数据划分：未改变，仍为当前12条 `train.clean.100` 小样本；没有读取或重分 dev/test。

验收指标：CLI 命令集合完全保留、公共产物序列化确定性、全部测试通过、安全重跑不改写既有产物。

停止条件：任一命令丢失、产物哈希变化、测试失败或安全重跑触发训练，均视为重构失败并停止下游工作。

状态：代码结构重构完成；模型指标和边界门禁结论不变。

### 步骤 24：整理 Git 分支、历史和大文件规则

当前阶段：仓库治理，不改变模型代码、数据划分、训练结果或边界门禁。

输入：本地和远端 `main`、`MFA`、`feature/hubert-phone-ctc`，当前重构提交，Git reflog 和对象库。

处理：

- 将项目结构重构保存为独立 `refactor` 提交。
- 新增顶层 `.gitignore`，停止跟踪 Python 字节码和模型权重类型。
- 创建并验证仓库外完整 bundle，备份活动分支和旧 Wav2Vec2 初始提交。
- 将 HuBERT 分支的三个已有提交重放到最新 `origin/main`。
- 发现 Windows CRLF 导致相同 YAML 的原始字节哈希变化。
- 新增 `.gitattributes`，固定配置、源码和文档为 LF，Windows 脚本为 CRLF，模型与音频为 binary。
- 使用 `--force-with-lease` 更新远端 HuBERT 分支。
- 将本地 `main` 以 `--ff-only` 同步到 `origin/main`。
- 将 MFA 提交归档为 `baseline-mfa-v1` annotated tag。
- 删除已经合并的本地和远端 `MFA` 分支。
- 在有完整备份后清理 reflog 和悬空对象。
- 新增 `docs/GIT_WORKFLOW.md` 固定后续分支生命周期。

最终引用：

```text
main -> e3d258a
feature/hubert-phone-ctc -> a88d4dc 及其后续文档提交
baseline-mfa-v1 -> d86f26f
```

验证结果：

```text
working tree: clean
local/remote branch tracking: synchronized
tests: 53/53 passed
compile: passed
data safe rerun: skipped
training safe rerun: skipped
tracked cache/model files: 0
git fsck: passed
Git object store: 285 MiB -> 125.50 KiB packed
```

数据划分：未改变。仍只有当前12条 `train.clean.100` 小样本，没有读取或重分 dev/test。

验收指标：分支图线性、`main` 同步、MFA 由标签归档、功能分支远端同步、大文件可恢复但不留在活动对象库、全部工程门禁通过。

停止条件：远端引用不一致、bundle 校验失败、测试失败、配置哈希变化或发现被跟踪的模型/缓存文件时，停止删除和对象清理。

状态：Git 仓库治理完成，当前继续在 `feature/hubert-phone-ctc` 开发。

## 5. 当前停止条件

以下事实禁止继续进入 Syllable CTC、VTL、API 或界面：

- 只有12条单说话人训练数据。
- 没有正式 dev-clean manifest。
- 没有多说话人泛化指标。
- CTC emission boundary MAE 约980 ms。
- 20/50 ms 边界 F1远未达到可用水平。

## 6. 下一步

下一阶段输入：多说话人的 `train-clean-100` 1至10小时子集，以及 `dev-clean` 对照子集。

下一阶段处理：

1. 生成无 MFA timestamp 的正式 phone-sequence manifest。
2. 只用 train 构建词表。
3. 冻结 HuBERT encoder 训练 Phone CTC head。
4. 逐步解冻顶部2至4层。
5. 使用 dev PER 选 checkpoint。
6. 使用 MFA 和人工复核子集评估边界。

下一阶段输出：多说话人 checkpoint、dev PER、20/50 ms 边界指标、错误样本清单和环境报告。

停止条件：如果1至10小时实验不能在 dev 上显著改善 PER 和边界指标，不扩大到完整 `train-clean-100`，先检查 blank 学习、静音处理、CTC 对齐语义和是否需要显式 Boundary head。
