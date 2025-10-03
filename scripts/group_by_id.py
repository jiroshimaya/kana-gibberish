#!/usr/bin/env python3
"""
output.jsonのdetailから同じidのものをグループ化するスクリプト
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def group_results_by_id(input_file: str, output_file: str) -> None:
    """
    output.jsonのdetailから同じidのものをグループ化してJSONファイルに出力

    Args:
        input_file: 入力JSONファイルパス
        output_file: 出力JSONファイルパス
    """
    logger.info(f"Reading input file: {input_file}")

    with Path(input_file).open(encoding="utf-8") as f:
        data = json.load(f)

    # idごとに結果をグループ化
    grouped_by_id: dict[str, dict[str, Any]] = {}

    # detailsを処理
    for detail in data.get("details", []):
        config_name = detail["config_name"]

        for result in detail.get("individual_results", []):
            result_id = result["id"]

            # 初回のidの場合、基本情報を設定
            if result_id not in grouped_by_id:
                grouped_by_id[result_id] = {
                    "id": result_id,
                    "wav_file": result["wav_file"],
                    "reference": result["reference"],
                    "results_by_config": {},
                }

            # 設定ごとの結果を追加
            grouped_by_id[result_id]["results_by_config"][config_name] = {
                "hypothesis": result["hypothesis"],
                "cer": result["cer"],
                "kana_distance": result["kana_distance"],
            }

    # 出力用データ構造を作成
    output_data = {
        "metadata": {
            "original_evaluation_date": data.get("evaluation_date"),
            "total_unique_samples": len(grouped_by_id),
            "total_configurations": len(data.get("details", [])),
            "grouping_date": "2025-10-02 00:15:00",  # 現在時刻
        },
        "grouped_results": list(grouped_by_id.values()),
    }

    num_details = len(data.get("details", []))
    logger.info(
        f"Grouped {len(grouped_by_id)} unique samples across "
        f"{num_details} configurations"
    )

    # 出力ファイルに書き込み
    logger.info(f"Writing output file: {output_file}")
    with Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("Grouping completed successfully")


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Group results by ID from output.json")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("output_file", help="Output JSON file path")

    args = parser.parse_args()

    # ファイルの存在確認
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    group_results_by_id(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
