# Wav2Vec2 LibriSpeech Fine-tuning

这是一个使用 `facebook/wav2vec2-base-960h` 在 LibriSpeech 数据集上进行 CTC
语音识别微调的最小示例。默认配置只训练 1 步，用于快速验证模型下载、音频预处理、
反向传播、评估和模型保存是否可以完整运行。

## 环境要求

- Python 3.10 或更高版本
- Windows、Linux 或 macOS
- CPU 可以完成默认冒烟测试；较大规模训练建议使用 CUDA GPU
- 首次运行需要连接 Hugging Face，以下载预训练模型和数据集

## 安装依赖

建议先创建并激活虚拟环境，然后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-wav2vec2.txt
```

Linux 或 macOS 使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-wav2vec2.txt
```

## 快速运行

执行默认的最小链路测试：

```powershell
python train_wav2vec2_librispeech.py
```

默认设置如下：

- 使用 LibriSpeech 极小测试数据集
- 2 条训练数据和 1 条验证数据
- 音频长度不超过 4 秒
- batch size 为 1
- 只训练 1 步
- 冻结 Wav2Vec2 特征编码器

成功运行后，终端会输出训练 loss、评估 loss、预测文本和参考文本。模型及处理器会保存到：

```text
outputs/wav2vec2-librispeech-smoke/
```

`outputs/` 包含体积较大的模型文件，已在 `.gitignore` 中排除，不应直接提交到 Git。

## 增加训练样本

GPU 环境下可以先进行一个小规模实验：

```powershell
python train_wav2vec2_librispeech.py `
  --train-samples 20 `
  --eval-samples 4 `
  --max-steps 20 `
  --batch-size 2 `
  --max-audio-seconds 10
```

使用 LibriSpeech `train-clean-100` 的部分数据：

```powershell
python train_wav2vec2_librispeech.py `
  --dataset full `
  --train-samples 96 `
  --eval-samples 4 `
  --max-steps 100 `
  --batch-size 2 `
  --max-audio-seconds 15 `
  --output-dir outputs/wav2vec2-librispeech-100
```

显存不足时可将 `--batch-size` 设置为 `1`。CPU 环境不建议直接进行较大规模微调。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-name` | `facebook/wav2vec2-base-960h` | Hugging Face CTC 模型名称或本地路径 |
| `--dataset` | `dummy` | 数据集类型，可选 `dummy` 或 `full` |
| `--train-samples` | `2` | 训练样本数 |
| `--eval-samples` | `1` | 验证样本数 |
| `--max-steps` | `1` | 最大训练步数 |
| `--batch-size` | `1` | batch size |
| `--learning-rate` | `1e-4` | AdamW 学习率 |
| `--max-audio-seconds` | `4.0` | 接受的最大音频时长；小于等于 0 表示不限制 |
| `--output-dir` | `outputs/wav2vec2-librispeech-smoke` | 模型输出目录 |
| `--seed` | `42` | 随机种子 |
| `--no-freeze-feature-encoder` | 未启用 | 训练时不冻结特征编码器 |

查看全部命令行参数：

```powershell
python train_wav2vec2_librispeech.py --help
```

## 项目结构

```text
.
|-- train_wav2vec2_librispeech.py  # 数据处理、训练、评估和保存入口
|-- requirements-wav2vec2.txt      # Python 依赖
|-- WAV2VEC2_FINETUNE.md            # 更详细的微调运行说明
|-- outputs/                        # 本地训练产物，不提交到 Git
`-- README.md
```

## 说明

默认任务只用于验证训练链路，并不代表模型经过了充分微调。要评估真实识别效果，应使用更大的
训练集、独立测试集和 WER 等指标。访问 Hugging Face Hub 频率较高时，可以设置 `HF_TOKEN`
以获得更高的下载限额。
