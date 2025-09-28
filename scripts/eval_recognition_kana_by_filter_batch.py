#!/usr/bin/env python3
"""
Kana ASR Evaluation Tool using filter mode with batch processing for JSON datasets.

Evaluates speech recognition models on JSON datasets containing audio files
and reference texts, calculating Character Error Rate (CER) for each sample.
Uses batch processing for improved GPU utilization and faster evaluation.

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Usage:
  python eval_recognition_kana_by_filter_batch.py dataset.json
  python eval_recognition_kana_by_filter_batch.py dataset.json --model reazon
  python eval_recognition_kana_by_filter_batch.py dataset.json --batch-size 8
  python eval_recognition_kana_by_filter_batch.py dataset.json \
    --wav-dir /path/to/wav/files
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from kanagib.evaluation import (
    evaluate_dataset_batch,
    load_json_dataset,
    print_evaluation_results,
)
from kanagib.speech_recognition import MODEL_CONFIGS, transcribe_audio_batch_with_filter

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="Kana ASR Evaluation Tool using filter mode with batch processing"
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
        "--batch-size",
        type=int,
        default=4,
        help="バッチサイズ (default: 4)",
    )
    ap.add_argument(
        "--wav-dir",
        help="音声ファイルディレクトリ"
        "（デフォルト: JSONファイルと同じディレクトリのwavフォルダ）",
    )
    args = ap.parse_args()

    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, model={args.model}, batch_size={args.batch_size}")
    
    # GPU情報を詳細に表示
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    json_path = Path(args.json_file)

    # JSONファイル存在確認
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    if json_path.suffix.lower() != ".json":
        logger.error(f"Input file must be a JSON file, got: {json_path.suffix}")
        sys.exit(1)

    try:
        logger.info("JSON dataset batch evaluation mode")

        # データセット読み込み
        dataset = load_json_dataset(str(json_path), args.wav_dir)

        if not dataset:
            logger.error("No valid data found in JSON file")
            sys.exit(1)

        # バッチ評価実行
        evaluation_results = evaluate_dataset_batch(
            dataset, 
            transcribe_audio_batch_with_filter, 
            batch_size=args.batch_size,
            model_name=args.model, 
            device=device
        )

        # 結果出力
        print_evaluation_results(evaluation_results, "filter (batch)")
        
        # 実行時間出力
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Average time per sample: {total_time/len(dataset):.2f} seconds")
        logger.info(f"Device used: {device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()