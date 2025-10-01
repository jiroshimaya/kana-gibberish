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
import json
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
from kanagib.schemas import (
    EvaluationBatch,
    EvaluationConfig,
    EvaluationResult,
    RecognitionMode,
)
from kanagib.speech_recognition import (
    transcribe_audio_batch_with_conversion,
    transcribe_audio_batch_with_filter,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> EvaluationBatch:
    """JSON設定ファイルを読み込んでEvaluationBatchを作成"""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_file.suffix.lower() != ".json":
        raise ValueError(
            f"Only JSON config files are supported, got: {config_file.suffix}"
        )

    with config_file.open(encoding="utf-8") as f:
        data = json.load(f)  # EvaluationBatchオブジェクト作成

    configs = []
    for config_data in data.get("configs", []):
        # JSONファイルパスをそのまま使用（相対パス対応）
        json_file = config_data["json_file"]

        # JSONファイルの存在チェック（相対パスで）
        if not Path(json_file).exists():
            raise FileNotFoundError(f"Dataset JSON file not found: {json_file}")

        # WAVディレクトリパスもそのまま使用
        wav_dir = config_data.get("wav_dir")

        config = EvaluationConfig(
            name=config_data["name"],
            json_file=json_file,
            model=config_data["model"],
            mode=RecognitionMode(config_data["mode"]),
            wav_dir=wav_dir,
            batch_size=config_data.get("batch_size", 4),
            use_batch=config_data.get("use_batch"),
            description=config_data.get("description", ""),
        )
        configs.append(config)

    return EvaluationBatch(
        name=data.get("name", "Evaluation Batch"),
        description=data.get("description", ""),
        configs=configs,
    )


def run_single_evaluation(
    config: EvaluationConfig, device: torch.device
) -> EvaluationResult:
    """単一の評価設定を実行"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting evaluation: {config.name}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Mode: {config.mode.value}")
    logger.info(f"JSON: {config.json_file}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'=' * 60}")

    start_time = time.time()

    # データセット読み込み
    logger.info(f"Loading dataset from: {config.json_file}")
    logger.info(f"WAV directory: {config.wav_dir}")

    dataset = load_json_dataset(config.json_file, config.wav_dir)
    if not dataset:
        raise ValueError(f"No valid data found in JSON file: {config.json_file}")

    logger.info(f"Loaded {len(dataset)} samples from dataset")

    # 常にバッチ処理を使用
    logger.info(f"Using batch processing (batch_size={config.batch_size})")

    # バッチ評価実行
    if config.mode == RecognitionMode.FILTER:
        batch_transcribe_func = transcribe_audio_batch_with_filter
        batch_info = f"filter (batch, size={config.batch_size})"
    else:  # CONVERSION
        batch_transcribe_func = transcribe_audio_batch_with_conversion
        batch_info = f"conversion (batch, size={config.batch_size})"

    evaluation_results = evaluate_dataset_batch(
        dataset,
        batch_transcribe_func,
        batch_size=config.batch_size,
        model_name=config.model,
        device=device,
    )

    # 結果出力
    print_evaluation_results(evaluation_results, batch_info)

    end_time = time.time()
    execution_time = end_time - start_time

    # 実行統計
    logger.info("\nExecution completed:")
    total_minutes = execution_time / 60
    logger.info(
        f"  Total time: {execution_time:.2f} seconds ({total_minutes:.2f} minutes)"
    )
    logger.info(
        f"  Average time per sample: {execution_time / len(dataset):.2f} seconds"
    )
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {config.batch_size}")

    # 結果オブジェクト作成
    return EvaluationResult(
        config_name=config.name,
        model=config.model,
        mode=config.mode,
        json_file=config.json_file,
        total_samples=len(dataset),
        average_cer=evaluation_results["average_cer"],
        execution_time=execution_time,
        device=str(device),
        batch_size=config.batch_size,
        details={
            "individual_results": evaluation_results["individual_results"],
            "total_errors": evaluation_results.get("total_errors", 0),
            "total_characters": evaluation_results.get("total_characters", 0),
        },
    )


def save_results(results: list[EvaluationResult], output_path: str) -> None:
    """結果をファイルに保存"""
    output_data = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluations": len(results),
        "results": [
            {
                "config_name": r.config_name,
                "model": r.model,
                "mode": r.mode.value,
                "json_file": r.json_file,
                "total_samples": r.total_samples,
                "average_cer": r.average_cer,
                "execution_time": r.execution_time,
                "device": r.device,
                "batch_size": r.batch_size,
            }
            for r in results
        ],
    }

    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


def print_summary(results: list[EvaluationResult]) -> None:
    """全体の実行結果サマリーを出力"""
    if not results:
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'=' * 80}")

    total_time = sum(r.execution_time for r in results)
    total_samples = sum(r.total_samples for r in results)

    logger.info(f"Total evaluations: {len(results)}")
    logger.info(f"Total samples processed: {total_samples}")
    total_minutes = total_time / 60
    logger.info(
        f"Total execution time: {total_time:.2f} seconds ({total_minutes:.2f} minutes)"
    )
    logger.info(f"Average time per evaluation: {total_time / len(results):.2f} seconds")

    logger.info("\nDetailed results:")
    for result in results:
        batch_info = f" (batch={result.batch_size})"
        logger.info(
            f"  {result.config_name}: "
            f"CER={result.average_cer:.4f}, "
            f"samples={result.total_samples}, "
            f"time={result.execution_time:.1f}s, "
            f"model={result.model}, "
            f"mode={result.mode.value}{batch_info}"
        )

    logger.info(f"{'=' * 80}")


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
        evaluation_batch = load_config_file(args.config_file)

        logger.info(f"\nEvaluation Batch: {evaluation_batch.name}")
        logger.info(f"Description: {evaluation_batch.description}")
        logger.info(f"Number of configurations: {len(evaluation_batch.configs)}")

        # 設定確認表示
        for i, config in enumerate(evaluation_batch.configs, 1):
            batch_info = f" (batch={config.batch_size})"

            logger.info(f"  {i}. {config.name}")
            logger.info(f"     Model: {config.model}")
            logger.info(f"     Mode: {config.mode.value}{batch_info}")
            logger.info(f"     JSON: {config.json_file}")
            if config.description:
                logger.info(f"     Description: {config.description}")

        if args.dry_run:
            logger.info("\nDry run mode - exiting without running evaluations")
            return

        # 評価実行
        results: list[EvaluationResult] = []

        for i, config in enumerate(evaluation_batch.configs, 1):
            logger.info(f"\nRunning evaluation {i}/{len(evaluation_batch.configs)}")

            try:
                result = run_single_evaluation(config, device)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to run evaluation '{config.name}': {e}")
                logger.exception("Full traceback:")
                continue

        # 全体サマリー出力
        print_summary(results)

        # 結果保存
        if args.output:
            save_results(results, args.output)

        if not results:
            logger.error("No evaluations completed successfully")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
