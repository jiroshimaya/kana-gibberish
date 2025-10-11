"""
evaluate_config関数のテスト

新しく追加されたevaluate_config関数の動作をテストする
"""

from kanagib.evaluation import evaluate_config
from kanagib.schemas import (
    DatasetItem,
    EvaluationConditionItem,
    EvaluationConfig,
)


class TestEvaluateConfig:
    """evaluate_config関数のテスト"""

    def test_正常系_基本処理(self):
        """基本的な設定評価処理"""
        # モックデータセット
        mock_dataset = [
            DatasetItem(
                id="test_001",
                text="テスト",
                wav_file="test.wav",
                wav_path="/path/to/test.wav",
            )
        ]

        # モック条件
        condition = EvaluationConditionItem(
            name="test_condition",
            description="Test condition",
            json_file="test.json",
            model="andrewmcdowell",  # 有効なモデル名を使用
            mode="filter",  # 有効なモード名を使用
            dataset=mock_dataset,
        )

        # モック設定
        config = EvaluationConfig(
            json_file="test_config.json",
            name="test_config",
            description="Test config",
            batch_size=1,
            conditions=[condition],
        )

        # モック転写関数
        def mock_transcribe_func(audio_paths):
            return ["テスト" for _ in audio_paths]

        transcribe_functions = [mock_transcribe_func]

        # テスト実行
        result = evaluate_config(config, transcribe_functions)

        # 結果検証
        assert result.metadata.config_name == "test_config"
        assert result.metadata.json_file == "test_config.json"
        assert result.metadata.total_samples == 1
        assert result.metadata.total_evaluations == 1
        assert len(result.summary) == 1
        assert result.summary[0].name == "test_condition"
        assert result.details is not None
        assert len(result.details) == 1
        first_detail = result.details[0]
        assert first_detail.id == "test_001"
        assert len(first_detail.results) == 1
        assert first_detail.results[0].distances["cer"] == 0.0

    def test_異常系_モデルが見つからない(self):
        """指定されたモデルに対応する転写関数が見つからない場合"""
        mock_dataset = [
            DatasetItem(
                id="test_001",
                text="テスト",
                wav_file="test.wav",
                wav_path="/path/to/test.wav",
            )
        ]

        condition = EvaluationConditionItem(
            name="test_condition",
            description="Test condition",
            json_file="test.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=mock_dataset,
        )

        config = EvaluationConfig(
            json_file="test_config.json",
            name="test_config",
            description="Test config",
            batch_size=1,
            conditions=[condition],
        )

        # 転写関数の数が不足しているケースを検証
        transcribe_functions: list = []

        # ValueErrorが発生することを確認
        try:
            evaluate_config(config, transcribe_functions)
            raise AssertionError("ValueErrorが発生すべき")
        except ValueError as e:
            assert "transcribe_functions length mismatch" in str(e)

    def test_正常系_複数条件(self):
        """複数条件の評価処理"""
        # 2つのモックデータセット
        mock_dataset1 = [
            DatasetItem(
                id="test_001",
                text="テストイチ",
                wav_file="test1.wav",
                wav_path="/path/to/test1.wav",
            )
        ]
        mock_dataset2 = [
            DatasetItem(
                id="test_002",
                text="テストニ",
                wav_file="test2.wav",
                wav_path="/path/to/test2.wav",
            )
        ]

        # 2つのモック条件
        condition1 = EvaluationConditionItem(
            name="condition1",
            description="First condition",
            json_file="test1.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=mock_dataset1,
        )
        condition2 = EvaluationConditionItem(
            name="condition2",
            description="Second condition",
            json_file="test2.json",
            model="reazon",
            mode="conversion",
            dataset=mock_dataset2,
        )

        config = EvaluationConfig(
            json_file="multi_test_config.json",
            name="multi_test_config",
            description="Multi condition test",
            batch_size=2,
            conditions=[condition1, condition2],
        )

        # 2つのモック転写関数
        def mock_transcribe_func1(audio_paths):
            return ["テストイチ" for _ in audio_paths]

        def mock_transcribe_func2(audio_paths):
            return ["テストニ" for _ in audio_paths]

        transcribe_functions = [mock_transcribe_func1, mock_transcribe_func2]

        # テスト実行
        result = evaluate_config(config, transcribe_functions)

        # 結果検証
        assert result.metadata.total_samples == 2  # 合計2サンプル
        assert result.metadata.total_evaluations == 2  # 2つの評価条件
        assert len(result.summary) == 2
        assert result.summary[0].name == "condition1"
        assert result.summary[1].name == "condition2"
        assert result.details is not None
        assert len(result.details) == 2
        assert {detail.id for detail in result.details} == {"test_001", "test_002"}

    def test_正常系_詳細なし(self):
        """詳細結果を含まない設定"""
        mock_dataset = [
            DatasetItem(
                id="test_001",
                text="テスト",
                wav_file="test.wav",
                wav_path="/path/to/test.wav",
            )
        ]

        condition = EvaluationConditionItem(
            name="test_condition",
            description="Test condition",
            json_file="test.json",
            model="andrewmcdowell",
            mode="filter",
            dataset=mock_dataset,
        )

        config = EvaluationConfig(
            json_file="test_config.json",
            name="test_config",
            description="Test config",
            batch_size=1,
            conditions=[condition],
        )

        def mock_transcribe_func(audio_paths):
            return ["テスト" for _ in audio_paths]

        transcribe_functions = [mock_transcribe_func]

        # 詳細なしでテスト実行
        result = evaluate_config(config, transcribe_functions)

        # 結果検証
        assert result.details is not None
        assert len(result.details) == 1
        assert len(result.summary) == 1
        assert result.metadata.total_samples == 1
