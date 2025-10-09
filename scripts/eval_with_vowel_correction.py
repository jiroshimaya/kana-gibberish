#!/usr/bin/env python3
"""
母音表記ゆれ補正後のCER・kana_distance計算スクリプト

evaluations/output.jsonを読み取り、リファレンスとhypothesisに母音表記ゆれ補正を適用後、
補正後のCERとkana_distanceを計算して結果を出力する。

母音補正ルール:
- 母音（あいうえお）の直前が同じ母音のカナのときその母音を伸ばし棒「ー」に変換
- 特殊ケース：
  - う: 直前がお（オ）のときも伸ばし棒に変換
  - い: 直前がえ（エ）のときも伸ばし棒に変換
"""

import json
import logging
from pathlib import Path
from typing import Any

import jamorasep

from kanagib.evaluation import calculate_cer, calculate_kana_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_vowel_notation(text: str) -> str:
    """母音表記ゆれを補正する

    Args:
        text: 補正対象のカタカナテキスト

    Returns:
        補正後のテキスト

    Examples:
        >>> normalize_vowel_notation("コーヒー")
        "コーヒー"
        >>> normalize_vowel_notation("コオヒイ")
        "コーヒー"
        >>> normalize_vowel_notation("トオキョオ")
        "トーキョー"
    """
    logger.debug(f"補正前: {text}")

    # 前処理：スペースを削除
    text = "".join(text.split())  # 半角・全角スペース削除

    mora_surfaces = jamorasep.parse(text, format="katakana")
    mora_consonants_vowels = jamorasep.parse(text, output_format="simple-ipa")

    result = mora_surfaces[0]

    for i in range(1, len(mora_surfaces)):
        current_surface = mora_surfaces[i]
        current_vowel = mora_consonants_vowels[i][-1]
        previous_vowel = mora_consonants_vowels[i - 1][-1]
        previous_surface = result[-1]

        if (
            previous_surface != "ー"
            and current_surface in "アイウエオァィゥェォ"
            and (
                previous_vowel == current_vowel
                or (previous_vowel == "o" and current_vowel == "u")
                or (previous_vowel == "e" and current_vowel == "i")
            )
        ):
            # 直前が伸ばし棒でない、かつ母音が同じ場合に伸ばし棒に変換
            result += "ー"
        else:
            result += current_surface
    logger.debug(f"補正後: {result}")
    return result


def normalize_kana(text: str) -> str:
    text = normalize_vowel_notation(text)
    text = text.replace("ヅ", "ズ").replace("ヂ", "ジ")
    return text


def process_evaluation_results(  # noqa: PLR0912, PLR0915
    input_file: Path, output_file: Path | None = None
) -> dict[str, Any]:
    """評価結果を母音補正付きで再計算

    Args:
        input_file: 入力JSONファイル（evaluations/output.json）
        output_file: 出力JSONファイル
            （Noneの場合は同じディレクトリに*_vowel_corrected.jsonを作成）

    Returns:
        補正結果の辞書
    """
    logger.info(f"処理開始: {input_file}")

    # 入力ファイル読み込み
    with input_file.open(encoding="utf-8") as file:
        data = json.load(file)

    # 出力ファイル名決定
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_vowel_corrected.json"

    # 元の構造を保持しつつ補正結果を追加
    corrected_data = {
        "evaluation_date": data["evaluation_date"],
        "total_evaluations": data["total_evaluations"],
        "summary": {
            "total_samples": data["summary"]["total_samples"],
            "total_execution_time": data["summary"]["total_execution_time"],
            "average_cer_across_configs": 0.0,  # 補正後の値で更新
            "average_kana_distance_across_configs": 0.0,  # 補正後の値で更新
            "original_average_cer_across_configs": data["summary"][
                "average_cer_across_configs"
            ],
            "original_average_kana_distance_across_configs": data["summary"][
                "average_kana_distance_across_configs"
            ],
            "cer_improvement": 0.0,
            "kana_distance_improvement": 0.0,
        },
        "results": [],
        "details": [],
    }

    # 新形式のdetailsを設定名ごとに分離してマッピングを作成
    details_by_config = {}
    if "details" in data:
        # 新形式の場合、各wavファイルのresults_by_configから設定ごとのリストを再構築
        for wav_detail in data["details"]:
            wav_id = wav_detail["id"]
            wav_file = wav_detail["wav_file"]
            reference = wav_detail["reference"]

            for config_name, config_result in wav_detail["results_by_config"].items():
                if config_name not in details_by_config:
                    details_by_config[config_name] = []

                details_by_config[config_name].append(
                    {
                        "id": wav_id,
                        "wav_file": wav_file,
                        "reference": reference,
                        "hypothesis": config_result["hypothesis"],
                        "cer": config_result["cer"],
                        "kana_distance": config_result["kana_distance"],
                    }
                )

    total_samples = 0
    total_cer = 0.0
    total_kana_distance = 0.0

    # 各設定の結果を処理
    for result in data["results"]:
        logger.info(f"処理中: {result['config_name']}")

        # 結果の基本構造をコピー
        corrected_result = {
            "config_name": result["config_name"],
            "model": result["model"],
            "mode": result["mode"],
            "json_file": result["json_file"],
            "total_samples": result["total_samples"],
            "average_cer": 0.0,  # 補正後の値
            "average_kana_distance": 0.0,  # 補正後の値
            "original_average_cer": result["average_cer"],
            "original_average_kana_distance": result["average_kana_distance"],
            "cer_improvement": 0.0,
            "kana_distance_improvement": 0.0,
            "execution_time": result["execution_time"],
            "device": result["device"],
        }

        # batch_sizeがある場合は追加
        if "batch_size" in result:
            corrected_result["batch_size"] = result["batch_size"]

        # 該当設定の詳細データがある場合は統計計算用の処理
        config_name = result["config_name"]

        if config_name in details_by_config:
            individual_results = details_by_config[config_name]
            sample_cer_sum = 0.0
            sample_kana_distance_sum = 0.0
            sample_count = 0

            for detail in individual_results:
                # 母音補正適用
                corrected_ref = normalize_kana(detail["reference"])
                corrected_hyp = normalize_kana(detail["hypothesis"])

                # 補正後のCER・kana_distance計算
                corrected_cer = calculate_cer(corrected_ref, corrected_hyp)
                corrected_kana_distance = calculate_kana_distance(
                    corrected_ref, corrected_hyp
                )

                sample_cer_sum += corrected_cer
                sample_kana_distance_sum += corrected_kana_distance
                sample_count += 1

            # 平均値計算
            if sample_count > 0:
                corrected_result["average_cer"] = sample_cer_sum / sample_count
                corrected_result["average_kana_distance"] = (
                    sample_kana_distance_sum / sample_count
                )
        else:
            # 詳細データがない場合は元の平均値をそのまま使用
            logger.warning(f"詳細データがありません: {result['config_name']}")
            corrected_result["average_cer"] = result["average_cer"]
            corrected_result["average_kana_distance"] = result["average_kana_distance"]

        # 改善度計算
        corrected_result["cer_improvement"] = (
            corrected_result["original_average_cer"] - corrected_result["average_cer"]
        )
        corrected_result["kana_distance_improvement"] = (
            corrected_result["original_average_kana_distance"]
            - corrected_result["average_kana_distance"]
        )

        # 全体統計用に累積
        total_samples += result["total_samples"]
        total_cer += corrected_result["average_cer"] * result["total_samples"]
        total_kana_distance += (
            corrected_result["average_kana_distance"] * result["total_samples"]
        )

        corrected_data["results"].append(corrected_result)

    # 新形式のdetailsを作成（wavファイルごとにグループ化）
    grouped_by_id = {}

    # 各設定の補正結果から、ID（wavファイル）ごとにグループ化
    for result in corrected_data["results"]:
        config_name = result["config_name"]

        if config_name in details_by_config:
            for detail in details_by_config[config_name]:
                # 補正処理
                corrected_ref = normalize_kana(detail["reference"])
                corrected_hyp = normalize_kana(detail["hypothesis"])

                corrected_cer = calculate_cer(corrected_ref, corrected_hyp)
                corrected_kana_distance = calculate_kana_distance(
                    corrected_ref, corrected_hyp
                )

                wav_id = detail["id"]

                if wav_id not in grouped_by_id:
                    grouped_by_id[wav_id] = {
                        "id": wav_id,
                        "wav_file": detail["wav_file"],
                        "reference": detail["reference"],
                        "corrected_reference": corrected_ref,
                        "results_by_config": {},
                    }

                grouped_by_id[wav_id]["results_by_config"][config_name] = {
                    "original_hypothesis": detail["hypothesis"],
                    "corrected_hypothesis": corrected_hyp,
                    "original_cer": detail["cer"],
                    "corrected_cer": corrected_cer,
                    "cer_improvement": detail["cer"] - corrected_cer,
                    "original_kana_distance": detail["kana_distance"],
                    "corrected_kana_distance": corrected_kana_distance,
                    "kana_distance_improvement": detail["kana_distance"]
                    - corrected_kana_distance,
                }

    # IDでソートしてリスト化
    corrected_data["details"] = [
        grouped_by_id[wav_id] for wav_id in sorted(grouped_by_id.keys())
    ]

    # 全体統計計算
    if total_samples > 0:
        corrected_data["summary"]["average_cer_across_configs"] = (
            total_cer / total_samples
        )
        corrected_data["summary"]["average_kana_distance_across_configs"] = (
            total_kana_distance / total_samples
        )
        corrected_data["summary"]["cer_improvement"] = (
            corrected_data["summary"]["original_average_cer_across_configs"]
            - corrected_data["summary"]["average_cer_across_configs"]
        )
        corrected_data["summary"]["kana_distance_improvement"] = (
            corrected_data["summary"]["original_average_kana_distance_across_configs"]
            - corrected_data["summary"]["average_kana_distance_across_configs"]
        )

    # 結果を出力
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(corrected_data, file, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存しました: {output_file}")

    # サマリー表示
    print("\n=== 母音表記ゆれ補正結果 ===")
    summary = corrected_data["summary"]
    print(f"処理サンプル数: {total_samples}")
    print("元の平均CER: " f"{summary['original_average_cer_across_configs']:.4f}")
    print("補正後平均CER: " f"{summary['average_cer_across_configs']:.4f}")
    print(f"CER改善度: {summary['cer_improvement']:.4f}")
    print(
        "元の平均kana_distance: "
        f"{summary['original_average_kana_distance_across_configs']:.4f}"
    )
    print(
        "補正後平均kana_distance: "
        f"{summary['average_kana_distance_across_configs']:.4f}"
    )
    print("kana_distance改善度: " f"{summary['kana_distance_improvement']:.4f}")

    return corrected_data


def main():
    """メイン処理"""
    input_file = Path("evaluations/output.json")

    if not input_file.exists():
        logger.error(f"入力ファイルが見つかりません: {input_file}")
        return

    try:
        process_evaluation_results(input_file)
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
