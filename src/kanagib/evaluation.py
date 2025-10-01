"""
評価処理の共通機能

このモジュールには音声認識の評価に関する共通機能を含む：
- CER（Character Error Rate）計算
- JSONデータセット処理
- バッチ評価処理
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Any

import editdistance
import jamorasep
from kanasim import create_kana_distance_calculator

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
        kanasim距離値 (0.0以上の値、エラー時は-1.0)
        ・ 0.0が完全一致
        ・ -1.0はエラー（非カタカナ文字の含有等）
        ・ referenceのmora数で正規化されている

    Note:
        kanasimはカタカナ専用ライブラリのため、カタカナ以外の文字
        （ASCII文字、数字、ひらがな、漢字、記号等）が含まれる場合は
        -1.0を返す。平均計算時にはこの値を除外すること

    Raises:
        なし（内部でエラーをキャッチして-1.0を返す）
    """
    try:
        # kanasim距離を計算
        calculator = get_kana_calculator()
        raw_distance = calculator.calculate(reference, hypothesis)

        # referenceのmora数を計算（"ー"を除く）
        reference_moras = jamorasep.parse(reference, output_format="katakana")
        mora_count = len([mora for mora in reference_moras if mora != "ー"])

        # mora数で正規化（0除算を防ぐ）
        if mora_count == 0:
            return 0.0 if raw_distance == 0.0 else -1.0

        normalized_distance = raw_distance / mora_count
        return normalized_distance

    except KeyError as e:
        # kanasimはカタカナ専用のため、カタカナ以外の文字でKeyErrorが発生
        logger.warning(
            f"kanasim calculation failed due to non-katakana characters "
            f"in '{reference}' vs '{hypothesis}': {e}. Returning error value."
        )
        return -1.0
    except Exception as e:
        # その他の予期しないエラー
        logger.error(
            f"Unexpected error in kanasim calculation for '{reference}' vs "
            f"'{hypothesis}': {e}. Returning error value."
        )
        return -1.0


def calculate_multiple_distances(reference: str, hypothesis: str) -> dict[str, float]:
    """複数の距離指標を一度に計算

    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト

    Returns:
        各距離指標の辞書 {'cer': float, 'kana_distance': float}
    """
    return {
        "cer": calculate_cer(reference, hypothesis),
        "kana_distance": calculate_kana_distance(reference, hypothesis),
    }


def load_json_dataset(
    json_path: str, wav_dir: str | None = None
) -> list[dict[str, Any]]:
    """JSONファイルからデータセットを読み込み

    Args:
        json_path: JSONファイルのパス
        wav_dir: 音声ファイルのディレクトリ
            （Noneの場合はJSONファイルと同じディレクトリのwavフォルダを使用）

    Returns:
        データセットのリスト

    Raises:
        FileNotFoundError: JSONファイルが見つからない場合
        json.JSONDecodeError: JSONの解析に失敗した場合
    """
    json_path_obj = Path(json_path)
    if not json_path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path_obj.open(encoding="utf-8") as f:
        data = json.load(f)

    # wav_dirが指定されていない場合のデフォルト設定
    if wav_dir is None:
        json_parent = json_path_obj.parent
        wav_dir_path = json_parent / "wav"
    else:
        wav_dir_path = Path(wav_dir)

    # 音声ファイルのパスを絶対パスに変換
    for item in data:
        wav_file = item["wav_file"]
        wav_path = wav_dir_path / wav_file

        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}")
            continue

        item["wav_path"] = str(wav_path)

    return data


def evaluate_dataset(
    dataset: list[dict[str, Any]], transcribe_func, **transcribe_kwargs
) -> dict[str, Any]:
    """データセット全体の音声認識評価を実行

    Args:
        dataset: 評価データセット
        transcribe_func: 転写関数（audio_path, **kwargs -> str）
        **transcribe_kwargs: 転写関数に渡すキーワード引数

    Returns:
        評価結果の辞書（individual_results, average_metrics等）
    """
    logger.info(f"Starting evaluation. Total samples: {len(dataset)}")

    results = []
    total_metrics = {"cer": 0.0, "kana_distance": 0.0}
    valid_metrics_count = {"cer": 0, "kana_distance": 0}
    # 統計計算用のデータ保存
    metrics_values = {"cer": [], "kana_distance": []}
    processed_count = 0

    for i, item in enumerate(dataset):
        if "wav_path" not in item:
            logger.warning(f"Skipping item {i}: no wav_path")
            continue

        wav_path = item["wav_path"]
        reference = item["text"]
        item_id = item.get("id", f"item_{i}")

        try:
            logger.info(f"Processing {i + 1}/{len(dataset)}: {item_id}")

            # 音声認識実行
            hypothesis = transcribe_func(wav_path, **transcribe_kwargs)

            # 複数距離指標を計算
            distances = calculate_multiple_distances(reference, hypothesis)

            result = {
                "id": item_id,
                "wav_file": item["wav_file"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": distances["cer"],
                "kana_distance": distances["kana_distance"],
            }

            results.append(result)

            # 各指標の合計を計算（エラー値-1.0を除外）
            for metric, value in distances.items():
                if metric == "kana_distance" and value == -1.0:
                    # kana_distanceのエラー値は除外
                    continue
                total_metrics[metric] += value
                valid_metrics_count[metric] += 1

            processed_count += 1

            # ログ出力（エラー値は適切に表示）
            cer_str = f"{distances['cer']:.4f}"
            kana_dist_str = (
                "ERROR"
                if distances["kana_distance"] == -1.0
                else f"{distances['kana_distance']:.4f}"
            )

            logger.info(f"  CER: {cer_str}")
            logger.info(f"  Kana Distance: {kana_dist_str}")
            logger.info(f"  REF: {reference}")
            logger.info(f"  HYP: {hypothesis}")

        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            continue

    # 結果まとめ（エラー値を除いた平均・最大・最小・標準偏差を計算）
    summary_metrics = {}
    if processed_count > 0:
        for metric, total_value in total_metrics.items():
            valid_count = valid_metrics_count[metric]
            values = metrics_values[metric]
            
            if valid_count > 0 and values:
                summary_metrics[f"average_{metric}"] = total_value / valid_count
                summary_metrics[f"max_{metric}"] = max(values)
                summary_metrics[f"min_{metric}"] = min(values)
                # 標準偏差の計算（サンプル数が1の場合は0.0）
                if len(values) > 1:
                    summary_metrics[f"sd_{metric}"] = statistics.stdev(values)
                else:
                    summary_metrics[f"sd_{metric}"] = 0.0
            else:
                summary_metrics[f"average_{metric}"] = 0.0
                summary_metrics[f"max_{metric}"] = 0.0
                summary_metrics[f"min_{metric}"] = 0.0
                summary_metrics[f"sd_{metric}"] = 0.0
    else:
        for metric in total_metrics:
            summary_metrics[f"average_{metric}"] = 0.0
            summary_metrics[f"max_{metric}"] = 0.0
            summary_metrics[f"min_{metric}"] = 0.0
            summary_metrics[f"sd_{metric}"] = 0.0

    return {
        "individual_results": results,
        **summary_metrics,  # average_, max_, min_, sd_のメトリクスを展開
        "processed_count": processed_count,
        "total_count": len(dataset),
        "valid_metrics_count": valid_metrics_count,  # 各指標の有効サンプル数
    }


def print_evaluation_results(
    evaluation_results: dict[str, Any], mode_description: str = ""
) -> None:
    """評価結果を標準出力に整形して表示

    Args:
        evaluation_results: evaluate_datasetの戻り値
        mode_description: モードの説明（例："filter", "conversion"等）
    """
    mode_suffix = f" ({mode_description})" if mode_description else ""

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS{mode_suffix}")
    print("=" * 60)

    processed = evaluation_results["processed_count"]
    total = evaluation_results["total_count"]
    print(f"\nProcessed: {processed}/{total} samples")

    # 有効サンプル数の表示
    if "valid_metrics_count" in evaluation_results:
        valid_counts = evaluation_results["valid_metrics_count"]
        print(f"Valid samples for CER: {valid_counts.get('cer', processed)}")
        kana_distance_count = valid_counts.get('kana_distance', processed)
        print(f"Valid samples for Kana Distance: {kana_distance_count}")

    # 複数の統計指標を表示
    print("\nMetrics Summary:")
    if "average_cer" in evaluation_results:
        avg_cer = evaluation_results['average_cer']
        max_cer = evaluation_results.get('max_cer', 0.0)
        min_cer = evaluation_results.get('min_cer', 0.0)
        sd_cer = evaluation_results.get('sd_cer', 0.0)
        print(f"  CER (Character Error Rate):")
        print(f"    Average: {avg_cer:.4f}")
        print(f"    Max: {max_cer:.4f}")
        print(f"    Min: {min_cer:.4f}")
        print(f"    SD: {sd_cer:.4f}")
    
    if "average_kana_distance" in evaluation_results:
        avg_kana = evaluation_results['average_kana_distance']
        max_kana = evaluation_results.get('max_kana_distance', 0.0)
        min_kana = evaluation_results.get('min_kana_distance', 0.0)
        sd_kana = evaluation_results.get('sd_kana_distance', 0.0)
        print(f"  Kana Distance (kanasim):")
        print(f"    Average: {avg_kana:.4f}")
        print(f"    Max: {max_kana:.4f}")
        print(f"    Min: {min_kana:.4f}")
        print(f"    SD: {sd_kana:.4f}")

    print("\n" + "-" * 60)
    print("Individual Results:")
    print("-" * 60)

    for result in evaluation_results["individual_results"]:
        print(f"\nID: {result['id']}")
        print(f"File: {result['wav_file']}")

        # 複数の距離指標を表示（エラー値は適切に表示）
        print("Metrics:")
        if "cer" in result:
            print(f"  CER: {result['cer']:.4f}")
        if "kana_distance" in result:
            if result["kana_distance"] == -1.0:
                print("  Kana Distance: ERROR (non-katakana characters)")
            else:
                print(f"  Kana Distance: {result['kana_distance']:.4f}")

        print(f"REF: {result['reference']}")
        print(f"HYP: {result['hypothesis']}")


def evaluate_dataset_batch(
    dataset: list[dict[str, Any]],
    batch_transcribe_func,
    batch_size: int = 4,
    **transcribe_kwargs,
) -> dict[str, Any]:
    """データセット全体の音声認識評価をバッチ処理で実行

    Args:
        dataset: 評価データセット
        batch_transcribe_func: バッチ転写関数（audio_paths, **kwargs -> list[str]）
        batch_size: バッチサイズ
        **transcribe_kwargs: 転写関数に渡すキーワード引数

    Returns:
        評価結果の辞書（individual_results, average_metrics等）
    """
    logger.info(
        f"Starting batch evaluation. Total samples: {len(dataset)}, "
        f"batch_size: {batch_size}"
    )

    # wav_pathが設定されたデータセットのみ処理
    valid_dataset = [item for item in dataset if "wav_path" in item]
    if len(valid_dataset) != len(dataset):
        logger.warning(
            f"Skipped {len(dataset) - len(valid_dataset)} items without wav_path"
        )

    if not valid_dataset:
        logger.error("No valid data found in dataset")
        return {
            "individual_results": [],
            "average_cer": 0.0,
            "average_kana_distance": 0.0,
            "max_cer": 0.0,
            "max_kana_distance": 0.0,
            "min_cer": 0.0,
            "min_kana_distance": 0.0,
            "sd_cer": 0.0,
            "sd_kana_distance": 0.0,
            "processed_count": 0,
            "total_count": len(dataset),
            "valid_metrics_count": {"cer": 0, "kana_distance": 0},
        }

    # バッチ処理用にパスを分離
    audio_paths = [item["wav_path"] for item in valid_dataset]

    try:
        # バッチ転写実行
        hypotheses = batch_transcribe_func(
            audio_paths, batch_size=batch_size, **transcribe_kwargs
        )

        if len(hypotheses) != len(valid_dataset):
            raise ValueError(
                f"Mismatch: {len(hypotheses)} results for {len(valid_dataset)} inputs"
            )

        # 結果を個別に評価
        results = []
        total_metrics = {"cer": 0.0, "kana_distance": 0.0}
        valid_metrics_count = {"cer": 0, "kana_distance": 0}
        # 統計計算用のデータ保存
        metrics_values = {"cer": [], "kana_distance": []}

        for i, (item, hypothesis) in enumerate(
            zip(valid_dataset, hypotheses, strict=False)
        ):
            reference = item["text"]
            item_id = item.get("id", f"item_{i}")

            # 複数距離指標を計算
            distances = calculate_multiple_distances(reference, hypothesis)

            result = {
                "id": item_id,
                "wav_file": item["wav_file"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": distances["cer"],
                "kana_distance": distances["kana_distance"],
            }

            results.append(result)

            # 各指標の合計を計算（エラー値-1.0を除外）
            for metric, value in distances.items():
                if metric == "kana_distance" and value == -1.0:
                    # kana_distanceのエラー値は除外
                    continue
                total_metrics[metric] += value
                valid_metrics_count[metric] += 1
                # 統計計算用にデータを保存
                metrics_values[metric].append(value)

            # ログ出力（エラー値は適切に表示）
            cer_str = f"{distances['cer']:.4f}"
            kana_dist_str = (
                "ERROR"
                if distances["kana_distance"] == -1.0
                else f"{distances['kana_distance']:.4f}"
            )

            logger.info(f"  CER: {cer_str}")
            logger.info(f"  Kana Distance: {kana_dist_str}")
            logger.info(f"  REF: {reference}")
            logger.info(f"  HYP: {hypothesis}")

        # 結果まとめ（エラー値を除いた平均・最大・最小・標準偏差を計算）
        processed_count = len(results)
        summary_metrics = {}
        if processed_count > 0:
            for metric, total_value in total_metrics.items():
                valid_count = valid_metrics_count[metric]
                values = metrics_values[metric]
                
                if valid_count > 0 and values:
                    summary_metrics[f"average_{metric}"] = total_value / valid_count
                    summary_metrics[f"max_{metric}"] = max(values)
                    summary_metrics[f"min_{metric}"] = min(values)
                    # 標準偏差の計算（サンプル数が1の場合は0.0）
                    if len(values) > 1:
                        summary_metrics[f"sd_{metric}"] = statistics.stdev(values)
                    else:
                        summary_metrics[f"sd_{metric}"] = 0.0
                else:
                    summary_metrics[f"average_{metric}"] = 0.0
                    summary_metrics[f"max_{metric}"] = 0.0
                    summary_metrics[f"min_{metric}"] = 0.0
                    summary_metrics[f"sd_{metric}"] = 0.0
        else:
            for metric in total_metrics:
                summary_metrics[f"average_{metric}"] = 0.0
                summary_metrics[f"max_{metric}"] = 0.0
                summary_metrics[f"min_{metric}"] = 0.0
                summary_metrics[f"sd_{metric}"] = 0.0

        return {
            "individual_results": results,
            **summary_metrics,  # average_, max_, min_, sd_のメトリクスを展開
            "processed_count": processed_count,
            "total_count": len(dataset),
            "valid_metrics_count": valid_metrics_count,  # 各指標の有効サンプル数
        }

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise
