"""Common schema definitions for the project."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RecognitionMode(str, Enum):
    """音声認識モード"""

    FILTER = "filter"
    CONVERSION = "conversion"


@dataclass
class EvaluationConfig:
    """評価設定"""

    name: str  # 評価設定の名前
    json_file: str  # JSONデータセットファイルパス
    model: str  # 使用するモデル
    mode: RecognitionMode  # 音声認識モード
    wav_dir: str | None = None  # 音声ファイルディレクトリ
    batch_size: int = 4  # バッチサイズ (GPU使用時はバッチ処理を自動適用)
    use_batch: bool | None = (
        None  # バッチ処理強制指定 (None=自動判定, True=強制バッチ, False=単体処理)
    )
    description: str = ""  # 評価の説明

    def __post_init__(self) -> None:
        """バリデーション"""
        if not Path(self.json_file).exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")

        if self.wav_dir and not Path(self.wav_dir).exists():
            raise FileNotFoundError(f"WAV directory not found: {self.wav_dir}")


@dataclass
class EvaluationBatch:
    """複数の評価設定をまとめたバッチ"""

    name: str  # バッチ名
    description: str  # バッチの説明
    configs: list[EvaluationConfig]  # 評価設定のリスト

    def __post_init__(self) -> None:
        """バリデーション"""
        if not self.configs:
            raise ValueError("At least one evaluation config is required")


@dataclass
class EvaluationResult:
    """評価結果"""

    config_name: str  # 評価設定名
    model: str  # 使用したモデル
    mode: RecognitionMode  # 使用した認識モード
    json_file: str  # 使用したJSONファイル
    total_samples: int  # 総サンプル数
    average_cer: float  # 平均CER
    execution_time: float  # 実行時間（秒）
    device: str  # 使用デバイス
    batch_size: int | None = None  # バッチサイズ（バッチ処理使用時）
    details: dict[str, Any] | None = None  # 詳細結果
