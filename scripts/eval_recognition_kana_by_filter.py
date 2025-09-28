#!/usr/bin/env python3
"""
Kana ASR Evaluation Tool using filter mode for JSON datasets.

Evaluates speech recognition models on JSON datasets containing audio files
and reference texts, calculating Character Error Rate (CER) for each sample.

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Usage:
  python eval_recognition_kana_by_filter.py dataset.json
  python eval_recognition_kana_by_filter.py dataset.json --model reazon
  python eval_recognition_kana_by_filter.py dataset.json \
    --wav-dir /path/to/wav/files
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from kanagib.evaluation import (
    evaluate_dataset,
    load_json_dataset,
    print_evaluation_results,
)
from kanagib.speech_recognition import MODEL_CONFIGS, transcribe_audio

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_func_filter(audio_path: str, **kwargs) -> str:
    """Filter方式での音声認識"""
    result = transcribe_audio(audio_path, mode="filter", **kwargs)
    # mode="filter"では常にstrが返される
    assert isinstance(result, str)
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Kana ASR Evaluation Tool using filter mode"
    )
    ap.add_argument(
        "json_file", help="JSONデータセットファイル（gibberish_with_wav.json形式）"
    )
    ap.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="andrewmcdowell",
        help="使用するモデル (default: andrewmcdowell)",
    )
    ap.add_argument(
        "--wav-dir",
        help="音声ファイルディレクトリ"
        "（デフォルト: JSONファイルと同じディレクトリのwavフォルダ）",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, model={args.model}")

    json_path = Path(args.json_file)

    # JSONファイル存在確認
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    if json_path.suffix.lower() != ".json":
        logger.error(f"Input file must be a JSON file, got: {json_path.suffix}")
        sys.exit(1)

    try:
        logger.info("JSON dataset evaluation mode")

        # データセット読み込み
        dataset = load_json_dataset(str(json_path), args.wav_dir)

        if not dataset:
            logger.error("No valid data found in JSON file")
            sys.exit(1)

        # バッチ評価実行
        evaluation_results = evaluate_dataset(
            dataset, transcribe_func_filter, model_name=args.model, device=device
        )

        # 結果出力
        print_evaluation_results(evaluation_results, "filter")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
