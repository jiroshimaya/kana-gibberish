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

logger = logging.getLogger(__name__)


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
        評価結果の辞書（individual_results, average_cer等）
    """
    logger.info(f"Starting evaluation. Total samples: {len(dataset)}")

    results = []
    total_cer = 0.0
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

            # CER計算
            cer = calculate_cer(reference, hypothesis)

            result = {
                "id": item_id,
                "wav_file": item["wav_file"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": cer,
            }

            results.append(result)
            total_cer += cer
            processed_count += 1

            logger.info(f"  CER: {cer:.4f}")
            logger.info(f"  REF: {reference}")
            logger.info(f"  HYP: {hypothesis}")

        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            continue

    # 結果まとめ
    average_cer = total_cer / processed_count if processed_count > 0 else 0.0

    return {
        "individual_results": results,
        "average_cer": average_cer,
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

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS{mode_suffix}")
    print("=" * 50)

    processed = evaluation_results["processed_count"]
    total = evaluation_results["total_count"]
    print(f"\nProcessed: {processed}/{total} samples")
    print(f"Average CER: {evaluation_results['average_cer']:.4f}")

    print("\n" + "-" * 50)
    print("Individual Results:")
    print("-" * 50)

    for result in evaluation_results["individual_results"]:
        print(f"\nID: {result['id']}")
        print(f"File: {result['wav_file']}")
        print(f"CER: {result['cer']:.4f}")
        print(f"REF: {result['reference']}")
        print(f"HYP: {result['hypothesis']}")
