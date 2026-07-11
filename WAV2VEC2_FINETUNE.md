# Wav2Vec2 + LibriSpeech 小样本微调

该流程使用 `facebook/wav2vec2-base-960h` 做英文 CTC 语音识别微调。默认数据是
LibriSpeech 的极小派生测试集，仅用于验证下载、音频处理、反向传播、评估和模型保存。

## 1. 安装依赖

```powershell
python -m pip install -r requirements-wav2vec2.txt
```

## 2. 最小链路测试

```powershell
python train_wav2vec2_librispeech.py
```

默认只使用 2 条训练数据、1 条验证数据，筛选不超过 4 秒的完整音频，训练 1 步。输出保存在
`outputs/wav2vec2-librispeech-smoke`。CPU 可以运行，但首次需要从 Hugging Face 下载模型和数据。

## 3. 增加少量样本

GPU 环境建议先运行：

```powershell
python train_wav2vec2_librispeech.py --train-samples 20 --eval-samples 4 --max-steps 20 --batch-size 2 --max-audio-seconds 10
```

确认小测试正常后，切换到 LibriSpeech `train-clean-100` 的前 100 条：

```powershell
python train_wav2vec2_librispeech.py --dataset full --train-samples 96 --eval-samples 4 --max-steps 100 --batch-size 2 --max-audio-seconds 15 --output-dir outputs/wav2vec2-librispeech-100
```

显存不足时将 `--batch-size` 改为 1；CPU 环境不建议直接进行较大规模微调。

也可以通过 `--model-name` 替换 Hugging Face 上兼容 CTC 的 Wav2Vec2 检查点。
