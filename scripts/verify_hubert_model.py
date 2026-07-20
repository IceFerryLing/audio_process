"""Run a deterministic offline HuBERT encoder smoke test on one second of audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import transformers
from transformers import AutoFeatureExtractor, HubertModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/models/facebook-hubert-base-ls960"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("data/samples/librispeech_train_clean_100/audio/374-180298-0000.flac"),
    )
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16_000 or waveform.ndim != 1:
        raise ValueError("HuBERT smoke-test audio must be mono 16 kHz")
    # 此烟雾测试必须证明固定的本地资产可用，不能再次访问网络补文件。
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = HubertModel.from_pretrained(args.model, local_files_only=True).eval()
    # 一秒音频足以低成本验证预处理和预期的49x768 encoder输出。
    inputs = feature_extractor(
        waveform[:sample_rate],
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
    print(
        json.dumps(
            {
                "status": "passed",
                "input_shape": list(inputs.input_values.shape),
                "last_hidden_state_shape": list(hidden.shape),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
