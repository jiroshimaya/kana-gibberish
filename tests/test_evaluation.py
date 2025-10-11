"""
評価機能のテスト

このモジュールには音声認識の評価機能に関するテストを含む：
- CER計算テスト
- kanasim距離計算テスト
- 複数距離計算テスト
- 評価データセット処理テスト
"""

# テストに必要な最小限のインポートのみ

from __future__ import annotations

from collections.abc import Callable as TypingCallable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest import CaptureFixture
else:
    Callable = TypingCallable

from kanagib.evaluation import (
    calculate_cer,
    calculate_distances,
    calculate_kana_distance,
    evaluate_condition_item,
    evaluate_config,
    print_condition_results,
    print_evaluation_result_summary,
)
from kanagib.schemas import (
    ConditionResultItem,
    DatasetItem,
    EvaluationConditionItem,
    EvaluationConfig,
    EvaluationMetadata,
    EvaluationResult,
    GroupedResultItem,
    RecognitionItem,
    SimpleRecognitionResultItem,
)


class TestCalculateCer:
    def test_正常系_完全一致(self) -> None:
        """完全一致の場合CERは0.0"""
        assert calculate_cer("テスト", "テスト") == 0.0

    def test_正常系_完全不一致(self) -> None:
        """完全に異なる場合CERは正の値"""
        cer = calculate_cer("アイウ", "カキク")
        assert cer > 0.0

    def test_エッジケース_参照が空文字列(self) -> None:
        """参照文字列が空の場合のCER計算"""
        assert calculate_cer("", "テスト") == 1.0

    def test_エッジケース_両方空文字列(self) -> None:
        """参照・仮説の両方が空文字列の場合"""
        assert calculate_cer("", "") == 0.0


class TestCalculateKanaDistance:
    def test_正常系_完全一致(self) -> None:
        """完全一致の場合距離は0.0"""
        assert calculate_kana_distance("テスト", "テスト") == 0.0

    def test_正常系_異なる文字(self) -> None:
        """異なる文字列間の距離は正の値"""
        distance = calculate_kana_distance("カナ", "ハナ")
        assert distance > 0.0


class TestCalculateDistances:
    def test_正常系_完全一致(self) -> None:
        """完全一致の場合複数距離が0.0になる"""
        distances = calculate_distances("テスト", "テスト")

        assert distances["cer"] == 0.0
        assert distances["kana_distance"] == 0.0

    def test_正常系_異なる文字(self) -> None:
        """異なる文字列の場合に距離が正しく計算される"""
        distances = calculate_distances("カナ", "ハナ")

        assert distances["cer"] > 0.0
        assert distances["kana_distance"] > 0.0


class TestEvaluateConditionItem:
    def test_正常系_基本処理(self) -> None:
        """条件評価が正しく実行される"""
        dataset = [
            DatasetItem(
                id="sample_001",
                text="テスト",
                wav_file="sample.wav",
                wav_path="/tmp/sample.wav",
            ),
            DatasetItem(
                id="sample_002",
                text="カナ",
                wav_file="sample2.wav",
                wav_path="/tmp/sample2.wav",
            ),
        ]

        condition_item = EvaluationConditionItem(
            name="demo_condition",
            description="Demo",
            json_file="demo.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=dataset,
        )

        def mock_transcribe(audio_paths: list[str]) -> list[str]:
            return ["テスト", "カナ"]

        condition_result, recognition_results = evaluate_condition_item(
            condition_item,
            mock_transcribe,
        )

        assert condition_result.name == "demo_condition"
        assert condition_result.total_samples == 2
        assert condition_result.distances["cer"] == 0.0
        assert condition_result.distances["kana_distance"] == 0.0

        assert len(recognition_results) == 2
        assert recognition_results[0].hypothesis == "テスト"
        assert recognition_results[1].hypothesis == "カナ"


class TestEvaluateConfig:
    def test_正常系_複数条件(self) -> None:
        """複数条件の設定を評価できる"""
        dataset1 = [
            DatasetItem(
                id="sample_001",
                text="テスト",
                wav_file="sample.wav",
                wav_path="/tmp/sample.wav",
            )
        ]
        dataset2 = [
            DatasetItem(
                id="sample_002",
                text="カナ",
                wav_file="sample2.wav",
                wav_path="/tmp/sample2.wav",
            )
        ]

        condition1 = EvaluationConditionItem(
            name="condition1",
            description="First",
            json_file="first.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=dataset1,
        )
        condition2 = EvaluationConditionItem(
            name="condition2",
            description="Second",
            json_file="second.json",
            model="reazon",
            mode="conversion",
            dataset=dataset2,
        )

        config = EvaluationConfig(
            name="evaluation_config",
            description="Test Config",
            json_file="config.json",
            batch_size=1,
            conditions=[condition1, condition2],
        )

        def transcribe_first(audio_paths: list[str]) -> list[str]:
            return ["テスト" for _ in audio_paths]

        def transcribe_second(audio_paths: list[str]) -> list[str]:
            return ["カナ" for _ in audio_paths]

        transcribe_functions: list[Callable[[list[str]], list[str]]] = [
            transcribe_first,
            transcribe_second,
        ]

        evaluation_result = evaluate_config(
            config,
            transcribe_functions,
        )

        metadata = evaluation_result.metadata
        assert metadata.config_name == "evaluation_config"
        assert metadata.json_file == "config.json"
        assert metadata.total_samples == 2
        assert metadata.total_evaluations == 2
        assert metadata.distances["cer"] == 0.0

        assert len(evaluation_result.summary) == 2
        assert evaluation_result.summary[0].name == "condition1"
        assert evaluation_result.summary[1].name == "condition2"

        assert evaluation_result.details is not None
        assert len(evaluation_result.details) == 2
        assert all(detail.results for detail in evaluation_result.details)

    def test_異常系_転写関数が足りない場合(self) -> None:
        """条件数より少ない転写関数の場合に適切に切り詰められる"""
        dataset = [
            DatasetItem(
                id="sample_001",
                text="テスト",
                wav_file="sample.wav",
                wav_path="/tmp/sample.wav",
            )
        ]

        condition = EvaluationConditionItem(
            name="condition",
            description="Condition",
            json_file="condition.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=dataset,
        )

        config = EvaluationConfig(
            name="evaluation_config",
            description="Test Config",
            json_file="config.json",
            batch_size=1,
            conditions=[condition],
        )

        def transcribe(audio_paths: list[str]) -> list[str]:
            return ["テスト" for _ in audio_paths]

        evaluation_result = evaluate_config(
            config,
            [transcribe],
        )

        assert evaluation_result.metadata.total_samples == 1
        assert len(evaluation_result.summary) == 1


class TestPrintConditionResults:
    def test_正常系_出力内容(self, capsys: CaptureFixture[str]) -> None:
        """print_condition_resultsの出力内容を検証"""
        condition_result = ConditionResultItem(
            name="condition",
            json_file="condition.json",
            total_samples=1,
            model="andrewmcdowell",
            mode="filter",
            distances={"cer": 0.0, "kana_distance": 0.0},
        )

        recognition_results = [
            RecognitionItem(
                id="sample_001",
                wav_file="sample.wav",
                reference="テスト",
                hypothesis="テスト",
                distances={"cer": 0.0, "kana_distance": 0.0},
            )
        ]

        print_condition_results(
            condition_result,
            recognition_results,
            mode_description="filter",
        )

        captured = capsys.readouterr()
        assert "EVALUATION RESULTS: condition (filter)" in captured.out
        assert "CER (Character Error Rate): 0.0000" in captured.out
        assert "Kana Distance (kanasim):    0.0000" in captured.out
        assert "REF: テスト" in captured.out
        assert "HYP: テスト" in captured.out


class TestPrintEvaluationResultSummary:
    def test_正常系_サマリー出力(self, capsys: CaptureFixture[str]) -> None:
        """print_evaluation_result_summaryの出力を検証"""
        metadata = EvaluationMetadata(
            config_name="evaluation_config",
            json_file="config.json",
            total_samples=2,
            distances={"cer": 0.1},
            execution_time=1.23,
            batch_size=2,
            evaluation_date="2025-10-09 10:00:00",
            total_evaluations=1,
        )

        summary = [
            ConditionResultItem(
                name="condition",
                json_file="condition.json",
                total_samples=2,
                model="andrewmcdowell",
                mode="filter",
                distances={"cer": 0.1},
            )
        ]

        details = [
            GroupedResultItem(
                id="sample_001",
                wav_file="sample.wav",
                reference="テスト",
                results=[
                    SimpleRecognitionResultItem(
                        condition_name="condition",
                        hypothesis="テスト",
                        distances={"cer": 0.1},
                    )
                ],
            )
        ]

        evaluation_result = EvaluationResult(
            metadata=metadata,
            summary=summary,
            details=details,
        )

        print_evaluation_result_summary(evaluation_result)

        captured = capsys.readouterr()
        assert "COMPREHENSIVE EVALUATION RESULTS" in captured.out
        assert "Total Samples: 2" in captured.out
        assert "condition" in captured.out
        assert "cer: 0.1000" in captured.out
