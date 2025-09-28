"""
評価処理の共通機能

このモジュールには音声認識の評価に関する共通機能を含む：
- CER（Character Error Rate）計算
- JSONデータセット処理
- バッチ評価処理
"""

import json
import logging
from pathlib import Path
from typing import Any

import editdistance
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


def calculate_kana_distance(reference: str, hypothesis: str) -> float:  # noqa: PLR0911
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
    try:
        calculator = get_kana_calculator()
        return calculator.calculate(reference, hypothesis)
    except KeyError as e:
        # kanasimはカタカナ専用のため、カタカナ以外の文字でKeyErrorが発生
        logger.warning(
            f"kanasim calculation failed due to non-katakana characters "
            f"in '{reference}' vs '{hypothesis}': {e}. Using fallback calculation."
        )

        # フォールバック: 編集距離をベースにした簡易的な類似度計算
        if reference == hypothesis:
            return 0.0

        # 編集距離を正規化した値をフォールバックとして使用
        edit_dist = editdistance.eval(reference, hypothesis)
        max_len = max(len(reference), len(hypothesis))
        if max_len == 0:
            return 0.0

        # 編集距離を10倍してkanasimの距離スケールに近づける
        return (edit_dist / max_len) * 10.0
    except Exception as e:
        # その他の予期しないエラー
        logger.error(
            f"Unexpected error in kanasim calculation for '{reference}' vs "
            f"'{hypothesis}': {e}. Using fallback calculation."
        )

        # 同じフォールバック処理
        if reference == hypothesis:
            return 0.0

        edit_dist = editdistance.eval(reference, hypothesis)
        max_len = max(len(reference), len(hypothesis))
        if max_len == 0:
            return 0.0

        return (edit_dist / max_len) * 10.0


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

            # 各指標の合計を計算
            for metric, value in distances.items():
                total_metrics[metric] += value

            processed_count += 1

            logger.info(f"  CER: {distances['cer']:.4f}")
            logger.info(f"  Kana Distance: {distances['kana_distance']:.4f}")
            logger.info(f"  REF: {reference}")
            logger.info(f"  HYP: {hypothesis}")

        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            continue

    # 結果まとめ
    average_metrics = {}
    if processed_count > 0:
        for metric, total in total_metrics.items():
            average_metrics[f"average_{metric}"] = total / processed_count
    else:
        for metric in total_metrics:
            average_metrics[f"average_{metric}"] = 0.0

    return {
        "individual_results": results,
        **average_metrics,  # average_cer, average_kana_distanceを展開
        "processed_count": processed_count,
        "total_count": len(dataset),
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

    # 複数の平均指標を表示
    print("\nAverage Metrics:")
    if "average_cer" in evaluation_results:
        print(f"  CER (Character Error Rate): {evaluation_results['average_cer']:.4f}")
    if "average_kana_distance" in evaluation_results:
        kana_dist = evaluation_results["average_kana_distance"]
        print(f"  Kana Distance (kanasim):    {kana_dist:.4f}")

    print("\n" + "-" * 60)
    print("Individual Results:")
    print("-" * 60)

    for result in evaluation_results["individual_results"]:
        print(f"\nID: {result['id']}")
        print(f"File: {result['wav_file']}")

        # 複数の距離指標を表示
        print("Metrics:")
        if "cer" in result:
            print(f"  CER: {result['cer']:.4f}")
        if "kana_distance" in result:
            print(f"  Kana Distance: {result['kana_distance']:.4f}")

        print(f"REF: {result['reference']}")
        print(f"HYP: {result['hypothesis']}")
