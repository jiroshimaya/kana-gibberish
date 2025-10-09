#!/usr/bin/env python3
"""
統合音声認識評価ツール

JSON設定ファイルから複数の評価条件を読み込んで、
順次実行する統合評価スクリプト。

特徴:
- 複数のモデル・データセット・認識モードを一括実行
- GPU環境では自動的にバッチ処理を適用
- 結果をまとめて出力・保存

使用例:
  python eval_recognition.py config.json
  python eval_recognition.py config.json --output results.json
  python eval_recognition.py config.json --dry-run
"""

import argparse
import logging
import sys

import torch

from kanagib.evaluation import (
    evaluate_config,
    print_evaluation_result_summary,
    save_evaluation_result,
)
from kanagib.schemas import (
    EvaluationConfig,
)
from kanagib.speech_recognition import (
    transcribe_audios,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="統合音声認識評価ツール - JSON設定ファイルから複数の評価を一括実行"
    )
    ap.add_argument(
        "config_file",
        help="評価設定ファイル (JSON形式)",
    )
    ap.add_argument(
        "--output",
        help="結果保存先ファイル (JSON形式)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="設定確認のみ実行（実際の評価は行わない）",
    )
    args = ap.parse_args()

    try:
        # デバイス確認
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {memory_gb:.1f} GB")

        # 設定ファイル読み込み
        logger.info(f"Loading config from: {args.config_file}")
        evaluation_config = EvaluationConfig.from_filepath(args.config_file)

        logger.info(f"\nEvaluation Config: {evaluation_config.name}")
        logger.info(f"Description: {evaluation_config.description}")
        logger.info(f"Number of conditions: {len(evaluation_config.conditions)}")

        # 設定確認表示
        for i, condition in enumerate(evaluation_config.conditions, 1):
            batch_info = f" (batch={evaluation_config.batch_size})"

            logger.info(f"  {i}. {condition.name}")
            logger.info(f"     Model: {condition.model}")
            logger.info(f"     Mode: {condition.mode}{batch_info}")
            logger.info(f"     JSON: {condition.json_file}")
            if condition.description:
                logger.info(f"     Description: {condition.description}")

        if args.dry_run:
            logger.info("\nDry run mode - exiting without running evaluations")
            return

        # 転写関数マップを作成（model_mode の組み合わせをキーとする）
        def create_transcribe_func(model_name: str, mode: str):
            """特定のモデルとモードに対応する転写関数を作成"""

            def transcribe_func(audio_paths: list[str]) -> list[str]:
                return transcribe_audios(
                    audio_paths=audio_paths,
                    mode=mode,
                    model_name=model_name,
                    device=device,
                    batch_size=evaluation_config.batch_size,
                )

            return transcribe_func

        # 条件ごとに転写関数リストを作成
        transcribe_functions = [
            create_transcribe_func(condition.model, condition.mode)
            for condition in evaluation_config.conditions
        ]

        # 評価実行
        evaluation_result = evaluate_config(
            config=evaluation_config,
            transcribe_functions=transcribe_functions,
        )

        # 結果表示
        print_evaluation_result_summary(evaluation_result)

        # 結果保存
        if args.output:
            save_evaluation_result(evaluation_result, args.output)
            logger.info(f"Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
