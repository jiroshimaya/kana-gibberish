"""
評価処理の共通機能

このモジュールには音声認識の評価に関する共通機能を含む：
- CER（Character Error Rate）計算
- JSONデータセット処理
- バッチ評価処理
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import editdistance
from kanasim import create_kana_distance_calculator

from kanagib.schemas import (
    ConditionResultItem,
    EvaluationConditionItem,
    EvaluationConfig,
    EvaluationMetadata,
    EvaluationResult,
    GroupedResultItem,
    RecognitionItem,
    SimpleRecognitionResultItem,
)

logger = logging.getLogger(__name__)

# kanasim距離計算器のグローバルインスタンス
_kana_calculator = None


def get_kana_calculator():
    """kanasim距離計算器のシングルトンインスタンスを取得"""
    global _kana_calculator  # noqa: PLW0603
    if _kana_calculator is None:
        _kana_calculator = create_kana_distance_calculator()
    return _kana_calculator


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER)を計算

    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト

    Returns:
        CER値 (0.0-1.0)
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    # 編集距離を計算
    edit_dist = editdistance.eval(reference, hypothesis)
    cer = edit_dist / len(reference)
    return cer


def calculate_kana_distance(reference: str, hypothesis: str) -> float:
    """kanasimを使用してカナ文字の音韻的距離を計算

    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト

    Returns:
        kanasim距離値 (0.0以上の値、0.0が完全一致)

    Note:
        kanasimはカタカナ専用ライブラリのため、カタカナ以外の文字
        （ASCII文字、数字、ひらがな、漢字、記号等）が含まれる場合は
        編集距離ベースのフォールバック値を返す

    Raises:
        なし（内部でKeyError等をキャッチしてフォールバック処理）
    """
    calculator = get_kana_calculator()
    return calculator.calculate(reference, hypothesis)


def calculate_distances(reference: str, hypothesis: str) -> dict[str, float]:
    """複数の距離指標を計算して辞書で返す

    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト

    Returns:
        dict[str, float]: 距離指標の辞書
            - "cer": Character Error Rate
            - "kana_distance": kanasim距離
    """
    return {
        "cer": calculate_cer(reference, hypothesis),
        "kana_distance": calculate_kana_distance(reference, hypothesis),
    }


def evaluate_condition_item(
    condition_item: EvaluationConditionItem,
    transcribe_func,
) -> tuple[ConditionResultItem, list[RecognitionItem]]:
    """データセット全体の音声認識をバッチ処理で評価実行（GPU最適化）

    Args:
        dataset: 評価データセット
        transcribe_batch_func: バッチ転写関数（audio_paths_list, **kwargs -> list[str]）
        batch_size: バッチサイズ（Noneの場合は関数内で自動調整）
        **transcribe_kwargs: 転写関数に渡すキーワード引数

    Returns:
        評価結果の辞書（individual_results, average_metrics等）
    """
    dataset = condition_item.dataset
    logger.info(f"Starting batch evaluation. Total samples: {len(dataset)}")

    # 音声パスリストを準備
    audio_paths = [item.wav_path for item in dataset]

    # バッチ転写を実行
    hypotheses = transcribe_func(audio_paths)

    # 結果を処理
    results = []

    for item, hypothesis in zip(dataset, hypotheses, strict=False):
        reference = item.text
        item_id = item.id

        distances = calculate_distances(reference, hypothesis)

        result = RecognitionItem(
            id=item_id,
            wav_file=item.wav_file,
            reference=reference,
            hypothesis=hypothesis,
            distances=distances,
        )

        results.append(result)

    distance_metrics = results[0].distances.keys()
    average_metrics = {
        metric: sum(r.distances[metric] for r in results) / len(results)
        for metric in distance_metrics
    }

    condition_result = ConditionResultItem(
        name=condition_item.name,
        json_file=condition_item.json_file,
        total_samples=len(results),
        distances=average_metrics,
        model=condition_item.model,
        mode=condition_item.mode,
    )
    return condition_result, results


def print_condition_results(
    condition_result: ConditionResultItem,
    recognition_results: list[RecognitionItem],
    mode_description: str = "",
) -> None:
    """条件別評価結果を標準出力に整形して表示

    Args:
        condition_result: 条件別評価結果
        recognition_results: 個別認識結果のリスト
        mode_description: モードの説明（例："filter", "conversion"等）
    """
    mode_suffix = f" ({mode_description})" if mode_description else ""

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {condition_result.name}{mode_suffix}")
    print("=" * 60)

    print(f"\nCondition: {condition_result.name}")
    print(f"Dataset: {condition_result.json_file}")
    print(f"Total samples: {condition_result.total_samples}")

    # 複数の平均指標を表示
    print("\nAverage Metrics:")
    for metric_name, value in condition_result.distances.items():
        if metric_name == "cer":
            print(f"  CER (Character Error Rate): {value:.4f}")
        elif metric_name == "kana_distance":
            print(f"  Kana Distance (kanasim):    {value:.4f}")
        else:
            print(f"  {metric_name}: {value:.4f}")

    print("\n" + "-" * 60)
    print("Individual Results:")
    print("-" * 60)

    for result in recognition_results:
        print(f"\nID: {result.id}")
        print(f"File: {result.wav_file}")

        # 複数の距離指標を表示
        print("Metrics:")
        for metric_name, value in result.distances.items():
            if metric_name == "cer":
                print(f"  CER: {value:.4f}")
            elif metric_name == "kana_distance":
                print(f"  Kana Distance: {value:.4f}")
            else:
                print(f"  {metric_name}: {value:.4f}")

        print(f"REF: {result.reference}")
        print(f"HYP: {result.hypothesis}")


def create_evaluation_result(
    condition_results: list[tuple[ConditionResultItem, list[RecognitionItem]]],
    execution_times: list[float],
    evaluation_config: EvaluationConfig,
) -> EvaluationResult:
    """複数の条件評価結果からEvaluationResultを生成

    Args:
        condition_results: (ConditionResultItem, RecognitionItem list)のペアのリスト
        execution_times: 各条件の実行時間のリスト
        device: 実行デバイス
        batch_size: バッチサイズ
        model: 使用モデル
        mode: 使用モード
        include_details: 詳細結果を含めるかどうか

    Returns:
        EvaluationResult: 統合された評価結果
    """
    evaluation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_evaluations = len(condition_results)

    # summaryはConditionResultItemのリスト
    summary = [condition_result for condition_result, _ in condition_results]

    # 全体統計を計算
    total_samples = sum(condition.total_samples for condition in summary)
    total_execution_time = sum(execution_times)

    # 平均距離指標を計算
    all_distances = [condition.distances for condition in summary]
    metric_names = set()
    for distances in all_distances:
        metric_names.update(distances.keys())

    avg_distances = {}
    for metric_name in metric_names:
        values = [d.get(metric_name, 0.0) for d in all_distances]
        avg_distances[metric_name] = sum(values) / len(values) if values else 0.0

    # metadataを作成（全体統計情報を含む）
    metadata = EvaluationMetadata(
        config_name=evaluation_config.name,
        json_file=evaluation_config.json_file,
        total_samples=total_samples,
        distances=avg_distances,
        execution_time=total_execution_time,
        evaluation_date=evaluation_date,
        total_evaluations=total_evaluations,
    )

    # 詳細結果の生成 - 全条件の結果をIDごとに統合
    grouped_by_id = {}

    # 全条件の結果を収集
    for condition_result, recognition_results in condition_results:
        for rec_result in recognition_results:
            if rec_result.id not in grouped_by_id:
                grouped_by_id[rec_result.id] = {
                    "id": rec_result.id,
                    "wav_file": rec_result.wav_file,
                    "reference": rec_result.reference,
                    "results": [],
                }

            simple_result = SimpleRecognitionResultItem(
                condition_name=condition_result.name,
                hypothesis=rec_result.hypothesis,
                distances=rec_result.distances,
            )
            grouped_by_id[rec_result.id]["results"].append(simple_result)

    # GroupedResultItemリストを作成
    details = []
    for grouped_data in grouped_by_id.values():
        grouped_item = GroupedResultItem(
            id=grouped_data["id"],
            wav_file=grouped_data["wav_file"],
            reference=grouped_data["reference"],
            results=grouped_data["results"],
        )
        details.append(grouped_item)

    return EvaluationResult(
        metadata=metadata,
        summary=summary,
        details=details,
    )


def save_evaluation_result(
    evaluation_result: EvaluationResult, output_path: str | Path
) -> None:
    """EvaluationResultをJSONファイルに保存

    Args:
        evaluation_result: 保存する評価結果
        output_path: 出力ファイルパス
    """
    evaluation_result.to_json(output_path)
    logger.info(f"Evaluation result saved to: {output_path}")


def evaluate_config(
    config: EvaluationConfig,
    transcribe_functions: list[Callable[[list[str]], list[str]]],
) -> EvaluationResult:
    """EvaluationConfigと転写関数のリストを使用して評価を実行し、EvaluationResultを返す

    Args:
        config: 評価設定
    transcribe_functions: 条件ごとに対応する転写関数のリスト

    Returns:
        EvaluationResult: 評価結果

    Raises:
        ValueError: 条件数と転写関数数が一致しない場合
    """
    logger.info(f"Starting config evaluation: {config.name}")
    logger.info(f"Total conditions: {len(config.conditions)}")

    if len(transcribe_functions) != len(config.conditions):
        msg = (
            "transcribe_functions length mismatch: "
            f"conditions={len(config.conditions)} functions={len(transcribe_functions)}"
        )
        raise ValueError(msg)

    condition_results = []
    execution_times = []

    for condition_item, transcribe_func in zip(
        config.conditions,
        transcribe_functions,
        strict=True,
    ):
        logger.info(f"Evaluating condition: {condition_item.name}")
        logger.info(f"Model: {condition_item.model}, Mode: {condition_item.mode}")

        # 実行時間を測定
        start_time = time.time()

        # 条件評価を実行
        condition_result, recognition_results = evaluate_condition_item(
            condition_item, transcribe_func
        )

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        condition_results.append((condition_result, recognition_results))

        logger.info(
            f"Condition '{condition_item.name}' completed in {execution_time:.2f}s"
        )

    # EvaluationResultを作成
    evaluation_result = create_evaluation_result(
        condition_results=condition_results,
        execution_times=execution_times,
        evaluation_config=config,
    )

    logger.info(f"Config evaluation completed: {config.name}")
    return evaluation_result


def print_evaluation_result_summary(evaluation_result: EvaluationResult) -> None:
    """EvaluationResultのサマリーを標準出力に表示

    Args:
        evaluation_result: 表示する評価結果
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    metadata = evaluation_result.metadata
    print(f"\nEvaluation Date: {metadata.evaluation_date}")
    print(f"Total Evaluations: {metadata.total_evaluations}")
    if metadata.batch_size:
        print(f"Batch Size: {metadata.batch_size}")

    print("\nOverall Statistics:")
    print(f"  Total Samples: {metadata.total_samples}")
    print(f"  Total Execution Time: {metadata.execution_time:.2f}s")
    print("  Overall Average Distances:")
    for metric_name, value in metadata.distances.items():
        print(f"    {metric_name}: {value:.4f}")

    print("\nCondition Results:")
    print("-" * 80)
    for condition in evaluation_result.summary:
        print(f"\nCondition: {condition.name}")
        print(f"  Dataset: {condition.json_file}")
        print(f"  Samples: {condition.total_samples}")
        print("  Distances:")
        for metric_name, value in condition.distances.items():
            print(f"    {metric_name}: {value:.4f}")
