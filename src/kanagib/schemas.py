"""Common schema definitions for the project."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypedDict


class DeviceInfo(TypedDict):
    """デバイス情報の型定義"""

    device_type: str
    device_name: str
    memory_total: int
    memory_available: int


@dataclass
class EvaluationConfig:
    name: str
    description: str
    json_file: str
    batch_size: int
    conditions: list[EvaluationConditionItem]

    @classmethod
    def from_filepath(cls, filepath: str) -> EvaluationConfig:
        """Load EvaluationConfig from a JSON file."""
        with Path(filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
        conditions = [
            EvaluationConditionItem.from_dict(item) for item in data["conditions"]
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            batch_size=data.get("batch_size", 1),
            json_file=filepath,
            conditions=conditions,
        )


@dataclass
class EvaluationConditionItem:
    name: str
    description: str
    json_file: str
    model: Literal["andrewmcdowell", "reazon"]
    mode: Literal["filter", "conversion"]
    dataset: list[DatasetItem]

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationConditionItem:
        """Create EvaluationConditionItem from a dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            json_file=data["json_file"],
            model=data["model"],
            mode=data["mode"],
            dataset=cls.load_dataset(data["json_file"]),
        )

    @classmethod
    def load_dataset(cls, json_path: str) -> list[DatasetItem]:
        """Load dataset items from a JSON file and optional WAV directory."""
        json_path_obj = Path(json_path)

        with json_path_obj.open(encoding="utf-8") as f:
            data = json.load(f)

        # If wav_dir is not specified, default to 'wav' folder in the same directory
        # as the JSON file
        wav_dir_path = json_path_obj.parent / "wav"

        dataset = []
        for item_dict in data:
            wav_file = item_dict["wav_file"]
            wav_path = wav_dir_path / wav_file

            dataset_item = DatasetItem(
                id=item_dict["id"],
                text=item_dict["text"],
                wav_file=item_dict["wav_file"],
                wav_path=str(wav_path),
            )
            dataset.append(dataset_item)

        return dataset


@dataclass
class DatasetItem:
    """データセットの個別アイテムの型定義"""

    id: str
    text: str
    wav_file: str
    wav_path: str


@dataclass
class RecognitionItem:
    """認識処理の個別アイテム"""

    id: str  # サンプルID
    wav_file: str  # 音声ファイルパス
    reference: str  # 正解テキスト
    hypothesis: str  # 認識結果テキスト
    distances: dict[str, float] = field(default_factory=dict)


@dataclass
class ConditionResultItem:
    name: str
    json_file: str
    total_samples: int
    model: str
    mode: str
    distances: dict[str, float]


@dataclass
class SimpleRecognitionResultItem:
    """グループ化された評価結果の個別アイテム"""

    condition_name: str
    hypothesis: str
    distances: dict[str, float]


@dataclass
class GroupedResultItem:
    """グループ化された評価結果の個別アイテム"""

    id: str
    wav_file: str
    reference: str
    results: list[SimpleRecognitionResultItem]


@dataclass
class EvaluationMetadata:
    config_name: str  # 評価設定名
    json_file: str  # 使用したJSONファイル
    total_samples: int  # 総サンプル数
    distances: dict[str, float]
    execution_time: float  # 実行時間（秒）
    batch_size: int | None = None  # バッチサイズ（バッチ処理使用時）
    evaluation_date: str | None = None  # 評価実行日時
    total_evaluations: int | None = None  # 評価設定数


@dataclass
class EvaluationResult:
    """評価結果の完全な構造"""

    metadata: EvaluationMetadata
    summary: list[ConditionResultItem]
    details: list[GroupedResultItem] | None = None  # 詳細結果（オプション）

    def to_json(self, filepath: str | Path) -> None:
        """JSON ファイルに出力"""

        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
