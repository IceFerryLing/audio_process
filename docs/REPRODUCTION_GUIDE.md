# 从零复现教程：一条命令一步解释

这份教程假设你对仓库、虚拟环境、HuBERT、MFA都不熟。不要一次复制整页命令。每完成一步，先检查“你应该看到什么”，确认成功后再继续。

仓库文件用途见 [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md)。当前研究结论和原理见 [`README.md`](../README.md)。

## 0. 你最终会复现什么

本教程复现当前小规模基线，不是正式大数据训练：

```text
固定 HuBERT Base
+ LibriSpeech train.clean.100 发布顺序前12条
+ 文本转ARPAbet标签
-> 冻结HuBERT的Phone CTC过拟合
-> Guided CTC音素对齐
-> 可选：MFA离线音节伪标签和边界对照
```

当前参考结果：

| 项目           |          参考值 |
| -------------- | --------------: |
| 音频           | 12条，168.625秒 |
| Phone标签      |          1537个 |
| Phone词表      |            55类 |
| 12条训练步数   |            2400 |
| 12条训练PER    |         约2.60% |
| 12条最终loss   |         约0.267 |
| MFA对照边界MAE |        约980 ms |
| 20 ms边界F1    |         约1.71% |
| 50 ms边界F1    |         约3.42% |

最后三项说明边界仍不可用。这是预期的失败门禁，不要为了得到“好看结果”修改报告。

## 1. 先认清两个命令窗口

教程中有两种代码块：

```powershell
# 这是Windows PowerShell命令
```

```bash
# 这是WSL Ubuntu里的bash命令
```

除MFA安装步骤外，所有命令都在Windows PowerShell执行。不要把PowerShell命令粘贴到Ubuntu，也不要把bash命令粘贴到PowerShell。

## 2. 准备电脑

最低要求：

- Windows 10或11；
- Git；
- Python 3.10以上，推荐Python 3.11；
- PowerShell；
- 网络连接；
- 至少约10 GB空闲空间，Python/PyTorch缓存可能继续占空间；
- 如果复现MFA，再准备WSL2 Ubuntu。

当前正确性训练使用CPU，不要求NVIDIA GPU。12条训练仍可能持续数分钟。

检查工具：

```powershell
git --version
python --version
$PSVersionTable.PSVersion
```

你应该看到Git版本、Python版本和PowerShell版本。若 `python` 提示找不到命令，先安装Python并勾选“Add Python to PATH”。

## 3. 克隆并切到正确分支

选择一个放项目的目录，例如桌面：

```powershell
Set-Location $HOME\Desktop
git clone https://github.com/IceFerryLing/audio_process.git
Set-Location audio_process
git switch feature/hubert-phone-ctc
```

为什么必须切分支：当前HuBERT Phone CTC、教程和最新结构在 `feature/hubert-phone-ctc`，默认 `main` 只到MFA合并节点。

如果你已经有仓库，不要重复clone：

```powershell
Set-Location C:\Users\LENOVN\Desktop\audio_process
git fetch --prune origin
git switch feature/hubert-phone-ctc
git pull --ff-only
```

确认位置：

```powershell
git status --short --branch
Test-Path pyproject.toml
Test-Path configs\train\hubert_phone_ctc_12_overfit.yaml
```

你应该看到：

```text
## feature/hubert-phone-ctc...origin/feature/hubert-phone-ctc
True
True
```

后续命令都必须从这个仓库根目录执行。

## 4. 创建独立Python环境

不要把项目依赖直接装进系统Python，也不要复用旧TensorFlow/Keras环境。

创建环境：

```powershell
python -m venv .venv
```

允许当前PowerShell进程执行激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

成功后，命令提示符前通常会出现 `(.venv)`。再确认Python确实来自仓库：

```powershell
python -c "import sys; print(sys.executable)"
```

输出路径应该包含：

```text
audio_process\.venv\Scripts\python.exe
```

升级pip并安装全部当前功能依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[download,test,review,model-check,train]"
```

这一步会安装PyTorch和Transformers，下载较大。等待命令返回，不要在安装中途关闭窗口。

检查关键包：

```powershell
python -c "import torch, transformers, soundfile, yaml; print(torch.__version__); print(transformers.__version__)"
```

当前参考版本：

```text
torch 2.9.1
transformers 5.13.0
```

最后确认CLI已经注册：

```powershell
sylrec --help
```

你应该看到 `data`、`train`、`align`、`evaluate` 四个命令组。

## 5. 先跑测试，不要急着训练

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前预期结尾：

```text
Ran 53 tests
OK
```

如果不是 `OK`，先停在这里。测试失败时继续下载、训练只会让问题更难判断。

## 6. 下载并校验HuBERT

执行：

```powershell
python scripts/download_hubert_model.py
```

脚本固定下载：

```text
facebook/hubert-base-ls960
revision: dba3bb02fda4248b6e082697eee756de8fe8aa8a
```

保存位置：

```text
assets/models/facebook-hubert-base-ls960/
```

关键文件：

```text
config.json
preprocessor_config.json
pytorch_model.bin
source.json
```

权重参考：

```text
大小：377,569,754 bytes
SHA-256：062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad
```

第一次成功会输出 `status: downloaded`。文件已经正确存在时输出 `status: skipped`，这也算成功。

不要随便加 `--overwrite`。只有明确知道本地文件残缺或哈希错误时才执行：

```powershell
python scripts/download_hubert_model.py --overwrite
```

## 7. 下载12条LibriSpeech

```powershell
python scripts/prepare_stage0_sample.py --count 12
```

数据固定来自：

```text
openslr/librispeech_asr
split: train.clean.100
发布顺序前12条
```

保存位置：

```text
data/samples/librispeech_train_clean_100/
|-- audio/
|   |-- 374-180298-0000.flac
|   `-- 其余11条FLAC
|-- manifest.jsonl
`-- source.json
```

检查数量：

```powershell
(Get-ChildItem data\samples\librispeech_train_clean_100\audio -Filter *.flac).Count
(Get-Content data\samples\librispeech_train_clean_100\manifest.jsonl).Count
```

两个命令都应该输出：

```text
12
```

已有正确数据时脚本返回 `skipped`。已有数据不匹配时脚本会停止，而不是静默覆盖。

### 7.1 下载Stage 3一小时多说话人子集

12条样本通过正确性门禁后，使用配置驱动的正式下载模式：

```powershell
sylrec data download `
  --config configs/data/librispeech_download_stage3.yaml `
  --split train-clean-100
```

同一个脚本也保留了兼容入口：

```powershell
python scripts/prepare_stage0_sample.py `
  --config configs/data/librispeech_download_stage3.yaml `
  --split train-clean-100
```

当前配置固定使用 `train.clean.100/0000.parquet`，按官方发布顺序扫描，每位说话人最多选择240秒，至少包含10位说话人，累计达到一小时后停止。它不会随机重分官方split。

输出位置：

```text
data/librispeech/train-clean-100/
|-- audio/
|-- manifest.jsonl
`-- source.json
```

当前实际结果：

```text
items: 289
duration: 3602.924812秒
speakers: 16
chapters: 20
```

机器可读总报告位于：

```text
artifacts/reports/librispeech_download_stage3.json
```

正确产物存在时重复执行会返回 `skipped`。需要完整官方split时，将对应配置的 `target_hours` 和 `target_items` 都设为 `null`，并将 `source_shards` 设为 `null` 以自动发现全部固定revision分片；完整训练仍必须等待一小时实验通过门禁。

## 8. 离线验证HuBERT前向

```powershell
python scripts/verify_hubert_model.py
```

脚本默认读取：

```text
data/samples/librispeech_train_clean_100/audio/374-180298-0000.flac
```

注意路径中有 `audio/` 子目录。缺少这一层会得到libsndfile无法打开文件的错误。

预期关键输出：

```json
{
  "status": "passed",
  "input_shape": [1, 16000],
  "last_hidden_state_shape": [1, 49, 768],
  "parameter_count": 94371712
}
```

这一步只证明HuBERT encoder能读取本地权重并前向，不会输出phones，也不会训练。

## 9. 构建不使用MFA时间戳的Phone标签

```powershell
sylrec data build-phone-sequences `
  --config configs/data/librispeech_phone_ctc_tiny.yaml
```

它做的事情：

```text
LibriSpeech文本
-> 规范化
-> CMUdict/OOV G2P
-> 有序ARPAbet phones
-> 训练集55类词表
-> CTC长度可行性检查
```

它不读取MFA TextGrid。

主要输出：

```text
artifacts/manifests/phone_ctc_tiny_train.jsonl
artifacts/vocabs/phone_ctc_tiny.json
artifacts/reports/phone_ctc_tiny_data.json
artifacts/reports/phone_ctc_tiny_data_config.yaml
```

查看报告：

```powershell
Get-Content artifacts\reports\phone_ctc_tiny_data.json
```

应该包含：

```text
items: 12
total phones: 1537
vocabulary size: 55
ctc infeasible items: 0
mfa timestamps used: false
```

正确产物已存在时返回 `skipped`。这表示哈希校验通过，不是命令没执行。

### 9.1 构建一小时Phone CTC标签

```powershell
sylrec data build-phone-sequences `
  --config configs/data/librispeech_phone_ctc_1h.yaml
```

输出：

```text
artifacts/manifests/phone_ctc_train_clean_100_1h.jsonl
artifacts/vocabs/phone_ctc_train_clean_100_1h.json
artifacts/reports/phone_ctc_train_clean_100_1h_data.json
```

当前结果：

```text
items: 289
duration: 3602.92481秒
phones: 36502
vocabulary size: 69
CTC infeasible items: 0
MFA timestamps used: false
unique OOV words: 179
```

OOV全部通过显式G2P生成标签，没有静默删除。正式扩大数据前仍需统计OOV token比例并抽查高频OOV发音。

## 10. 按顺序执行三次训练门禁

训练必须从最小风险开始，不要直接跳到12条。

### 10.1 一步smoke test

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_tiny_smoke.yaml
```

目标：真实HuBERT前向、CTC loss、反向传播和checkpoint都能运行。它只训练一步，不要求PER很低。

输出目录：

```text
runs/hubert-phone-ctc-tiny-smoke/
```

### 10.2 一条音频过拟合

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_tiny_overfit.yaml
```

参考结果：600步，最终loss约0.0038，训练PER为0。它只证明训练实现正确，不证明泛化。

输出目录：

```text
runs/hubert-phone-ctc-tiny-overfit/
```

### 10.3 十二条音频过拟合

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml
```

配置会：

- 冻结全部HuBERT encoder；
- 先缓存12条音频的HuBERT hidden states；
- 只训练42,295个Phone CTC head参数；
- 运行2400步；
- 保存best和last checkpoint；
- 恢复checkpoint做检查；
- 计算训练集loss和PER。

当前参考结果：

```text
status: passed_overfit
steps: 2400
final train loss: 约0.2667
final train PER: 约0.0260，也就是2.60%
checkpoint restore: true
```

验收门槛来自配置：PER不高于5%，最终loss不高于1。

查看指标：

```powershell
Get-Content runs\hubert-phone-ctc-12-overfit\metrics.json
```

检查checkpoint是否存在：

```powershell
Test-Path runs\hubert-phone-ctc-12-overfit\best\checkpoint.pt
Test-Path runs\hubert-phone-ctc-12-overfit\last\checkpoint.pt
```

两个结果都应该是 `True`。

### 10.4 一小时冻结encoder训练

一小时manifest通过后运行：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_1h_frozen.yaml
```

配置使用289条音频、冻结整个HuBERT encoder、缓存hidden states，只训练Phone CTC分类头，共5个epoch、最多1445步，输出到：

```text
runs/hubert-phone-ctc-1h-frozen/
```

当前配置使用CPU和batch size 1，首次缓存一小时HuBERT特征可能耗时较长。这个实验只验证多说话人数据上的训练收敛、有限loss和checkpoint恢复；由于尚无 `dev-clean`，不得用训练PER选择正式模型或声称泛化能力。

### 10.5 为什么命令可能直接显示skipped

如果metrics、配置、manifest和词表哈希全部一致，训练命令会返回：

```json
{"status": "skipped"}
```

这表示安全重跑生效。要明确重新初始化并覆盖当前正确性实验，才使用：

```powershell
sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml `
  --overwrite
```

`--overwrite` 会改写该run的checkpoint和metrics。使用前先确认你不需要旧结果。

固定seed只能保证门禁指标接近复现；当前Windows CPU多线程训练尚不保证checkpoint逐字节相同。

## 11. 运行Guided CTC音素对齐

先从manifest读取第一条真实文本，避免手工复制错误：

```powershell
$row = Get-Content artifacts\manifests\phone_ctc_tiny_train.jsonl |
  Select-Object -First 1 |
  ConvertFrom-Json

$row.id
$row.text
```

执行对齐：

```powershell
sylrec align phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml `
  --checkpoint runs/hubert-phone-ctc-12-overfit/best/checkpoint.pt `
  --audio data/samples/librispeech_train_clean_100/audio/374-180298-0000.flac `
  --text "$($row.text)"
```

输出包含：

- `target_phones`：目标文本得到的ARPAbet；
- `greedy_phones`：CTC直接贪心解码；
- `segments`：每个目标phone的start/end/confidence；
- `timestamp_semantics: ctc_emission_span`；
- `mfa_timestamps_used: false`。

这里的start/end是CTC发射区间，不是精确音素持续时间。

到这里已经完成不依赖MFA训练时间戳的HuBERT Phone CTC主线。如果只想复现HuBERT部分，可以跳到第18节做最终检查。

## 12. 可选：准备WSL2 MFA环境

MFA只用于离线伪标签和边界评价。训练Phone CTC不要求先完成MFA。

检查WSL：

```powershell
wsl --status
wsl -l -v
```

没有Ubuntu时，以管理员PowerShell执行：

```powershell
wsl --install -d Ubuntu
```

根据系统提示重启，然后第一次进入Ubuntu并创建Linux用户名和密码：

```powershell
wsl -d Ubuntu
```

下面开始是Ubuntu bash命令。

安装micromamba：

```bash
mkdir -p "$HOME/.local/bin"
workdir="$(mktemp -d)"
cd "$workdir"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj bin/micromamba
install -m 0755 bin/micromamba "$HOME/.local/bin/micromamba"
"$HOME/.local/bin/micromamba" --version
```

建立独立MFA CPU环境：

```bash
"$HOME/.local/bin/micromamba" create -y \
  -p "$HOME/.local/share/mfa" \
  -c conda-forge \
  python=3.11 \
  montreal-forced-aligner=3.3.7 \
  'kaldi=5.5.1172=cpu*'
```

验证MFA：

```bash
"$HOME/.local/bin/micromamba" run \
  -p "$HOME/.local/share/mfa" \
  mfa version
```

预期版本：

```text
3.3.7
```

输入 `exit` 回到Windows PowerShell：

```bash
exit
```

## 13. 准备MFA corpus和词典

回到仓库根目录的Windows PowerShell：

```powershell
sylrec data prepare-mfa `
  --config configs/data/librispeech_stage2.yaml
```

它会生成：

```text
artifacts/mfa/stage2/corpus/
artifacts/mfa/stage2/lexicon.txt
artifacts/manifests/stage2_prepared_train.jsonl
artifacts/reports/stage2_prepare.json
artifacts/reports/stage2_oov.jsonl
```

查看报告：

```powershell
Get-Content artifacts\reports\stage2_prepare.json
```

这一步有phones和音节结构，但还没有声学时间戳。

## 14. 下载MFA声学模型

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/download_mfa_model.ps1
```

固定模型：

```text
english_us_arpa v3.0.0
大小：91,928,208 bytes
SHA-256：d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b
```

保存位置：

```text
artifacts/mfa/models/english_us_arpa-v3.0.0.zip
```

脚本会并行下载分片并校验最终哈希。正确文件存在时输出 `skipped`。

## 15. 运行MFA validate和align

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_mfa_stage2.ps1
```

wrapper依次执行：

```text
mfa validate
-> 只有成功才继续
mfa align
-> mfa version
```

检查TextGrid数量：

```powershell
(Get-ChildItem artifacts\mfa\stage2\aligned -Recurse -Filter *.TextGrid).Count
```

预期：

```text
12
```

已有完整12个TextGrid时wrapper返回 `skipped`。

## 16. 构建带时间戳的MFA音节manifest

```powershell
sylrec data build `
  --config configs/data/librispeech_stage2.yaml `
  --mfa-version 3.3.7
```

预期主要结果：

```text
输入：12
成功：12
失败：0
words：452
phones：1537
syllables：619
```

输出：

```text
artifacts/manifests/stage2_aligned_train.jsonl
artifacts/reports/stage2_items.jsonl
artifacts/reports/stage2_build.json
artifacts/reports/stage2_manual_review.jsonl
```

## 17. 人工审核：不能机械点击通过

先生成10条审核图：

```powershell
python scripts/render_stage2_review.py `
  --manifest artifacts/manifests/stage2_aligned_train.jsonl `
  --output-dir artifacts/reports/stage2_review_visuals `
  --count 10
```

打开输出目录：

```powershell
Invoke-Item artifacts\reports\stage2_review_visuals
```

逐张检查：

- 波形是否存在；
- word边界是否大致覆盖对应单词；
- syllable边界是否单调、不重叠；
- ARPAbet标签是否和文本相符；
- 是否有明显长静音被分进音节。

只有实际检查后才能记录pass。把占位符替换为真实信息：

```powershell
sylrec data record-review `
  --config configs/data/librispeech_stage2.yaml `
  --all-items `
  --status pass `
  --reviewer "你的名字" `
  --method waveform-word-syllable-arpabet-visual `
  --evidence artifacts/reports/stage2_review_visuals/stage2_review_sheet_1.png `
  --evidence artifacts/reports/stage2_review_visuals/stage2_review_sheet_2.png `
  --notes "我逐条检查了波形、word、syllable和ARPAbet标签"
```

如果发现问题，必须使用 `--status fail` 并写清原因，不能为了打开门禁填写pass。

执行最终数据门禁：

```powershell
sylrec data validate `
  --config configs/data/librispeech_stage2.yaml
```

通过时应该包含：

```text
status: passed
problems: []
manual_gate_passed: true
downstream_training_allowed: true
```

pending或fail都会让命令返回失败。这是正确行为。

## 18. 使用MFA评价新Phone CTC checkpoint

```powershell
sylrec evaluate phone-ctc-mfa `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml `
  --checkpoint runs/hubert-phone-ctc-12-overfit/best/checkpoint.pt `
  --phone-manifest artifacts/manifests/phone_ctc_tiny_train.jsonl `
  --mfa-manifest artifacts/manifests/stage2_aligned_train.jsonl `
  --item-id 374-180298-0000 `
  --output artifacts/reports/hubert_phone_ctc_12_overfit_vs_mfa.json
```

查看报告：

```powershell
Get-Content artifacts\reports\hubert_phone_ctc_12_overfit_vs_mfa.json
```

当前参考结果：

```text
phone count: 117
paired boundaries: 234
boundary MAE: 约979.74 ms
20 ms F1: 约1.71%
50 ms F1: 约3.42%
mfa timestamps used for training: false
```

评价中的MFA只提供参考边界，不参与Phone CTC训练。

## 19. 最终完整检查

再次运行全部测试：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

编译Python文件：

```powershell
python -m compileall -q src tests scripts
```

检查Git没有把本地产物当成待提交文件：

```powershell
git status --short --branch
```

正常情况下只显示分支行，不显示 `assets/`、`data/`、`artifacts/` 或 `runs/`。

检查安全重跑：

```powershell
sylrec data build-phone-sequences `
  --config configs/data/librispeech_phone_ctc_tiny.yaml

sylrec train phone-ctc `
  --config configs/train/hubert_phone_ctc_12_overfit.yaml
```

两个命令都应该返回 `status: skipped`。

## 20. 常见错误怎么处理

### 20.1 `sylrec` 不是命令

原因：虚拟环境没激活，或者项目没有editable install。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[download,test,review,model-check,train]"
sylrec --help
```

### 20.2 `ModuleNotFoundError`

先确认当前Python：

```powershell
python -c "import sys; print(sys.executable)"
```

如果路径不在 `.venv`，重新激活环境。不要逐个在全局Python里补包。

### 20.3 下载脚本提示existing files mismatch

脚本发现已有文件但大小或哈希不正确。先查看文件是否是中断下载，不要直接删除整个仓库。确认后才使用对应的 `--overwrite` 或 `-Overwrite`。

### 20.4 `FileExistsError: existing ... outputs do not match`

说明输入、配置或产物哈希发生变化。这是安全门禁。依次检查：

```powershell
git status --short
Get-FileHash configs\data\librispeech_phone_ctc_tiny.yaml -Algorithm SHA256
Get-Content artifacts\reports\phone_ctc_tiny_data.json
```

不要把 `--overwrite` 当成通用修复按钮。

### 20.5 libsndfile打不开音频

确认真实路径包含 `audio` 子目录：

```powershell
Test-Path data\samples\librispeech_train_clean_100\audio\374-180298-0000.flac
```

应该输出 `True`。

### 20.6 `CUDA was requested but is unavailable`

当前三个正确性配置都使用CPU。检查配置中的：

```yaml
training:
  device: cpu
```

不要在没有CUDA环境时改成 `cuda`。

### 20.7 WSL找不到Ubuntu

```powershell
wsl -l -v
```

确保列表中有Ubuntu且版本为2。wrapper当前使用发行版名 `ubuntu`。

### 20.8 MFA命令找不到Kaldi工具

不要直接运行环境目录里的 `mfa`。使用：

```bash
"$HOME/.local/bin/micromamba" run \
  -p "$HOME/.local/share/mfa" \
  mfa version
```

项目wrapper也使用这种方式。

### 20.9 训练直接返回skipped

这不是失败，而是现有run和配置/数据哈希一致。只有明确要求重新训练时才增加 `--overwrite`。

### 20.10 训练PER不错但边界非常差

这是当前已知结果。CTC学习的是有序label路径，spike位置不等于完整phone持续区间。下一阶段需要多说话人数据、dev-clean和显式Boundary head，不能通过调报告数字解决。

## 21. 怎么判断你真的复现成功

最低检查清单：

```text
[ ] 位于 feature/hubert-phone-ctc
[ ] 使用独立 .venv
[ ] 53项测试通过
[ ] HuBERT文件大小和SHA-256通过
[ ] 12条FLAC和12行原始manifest存在
[ ] HuBERT smoke test输出 [1,49,768]
[ ] Phone manifest有12条、1537 phones、55类词表
[ ] 12条训练完成2400步
[ ] final PER不高于5%
[ ] final loss不高于1
[ ] best/last checkpoint存在且可恢复
[ ] 再次训练返回skipped
[ ] MFA没有进入Phone CTC训练监督
[ ] 边界失败结果被如实保留
```

## 22. 复现完成后下一步是什么

不要继续反复训练这12条单说话人音频。正确下一步是：

1. 下载多说话人的1至10小时 `train-clean-100` 子集；
2. 下载独立 `dev-clean` 子集；
3. 只从train构建词表；
4. 先冻结encoder训练Phone CTC head；
5. 再逐步解冻HuBERT顶部2至4层；
6. 使用dev PER选择checkpoint；
7. 使用MFA和人工复核子集评价边界；
8. 边界仍失败时实现显式Boundary head，而不是直接进入Syllable CTC、VTL或前端。
