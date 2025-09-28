"""
評価機能のテスト

このモジュールには音声認識の評価機能に関するテストを含む：
- CER計算テスト
- kanasim距離計算テスト
- 複数距離計算テスト
- 評価データセット処理テスト
"""

# テストに必要な最小限のインポートのみ

from kanagib.evaluation import (
    calculate_cer,
    calculate_kana_distance,
    calculate_multiple_distances,
    evaluate_dataset,
    print_evaluation_results,
)


class TestCalculateCer:
    def test_正常系_完全一致(self):
        """完全一致の場合CERは0.0"""
        assert calculate_cer("テスト", "テスト") == 0.0

    def test_正常系_完全不一致(self):
        """完全に異なる場合CERは1.0以上"""
        cer = calculate_cer("アイウ", "カキク")
        assert cer > 0.0

    def test_正常系_部分一致(self):
        """部分的に一致する場合の計算"""
        cer = calculate_cer("アイウエオ", "アイカエオ")
        # 5文字中1文字違うので 1/5 = 0.2
        assert cer == 0.2

    def test_エッジケース_空文字列_reference(self):
        """参照が空文字列の場合"""
        assert calculate_cer("", "テスト") == 1.0

    def test_エッジケース_空文字列_hypothesis(self):
        """仮説が空文字列の場合"""
        assert calculate_cer("テスト", "") == 1.0

    def test_エッジケース_両方空文字列(self):
        """両方が空文字列の場合"""
        assert calculate_cer("", "") == 0.0


class TestCalculateKanaDistance:
    def test_正常系_完全一致(self):
        """完全一致の場合距離は0.0"""
        assert calculate_kana_distance("テスト", "テスト") == 0.0

    def test_正常系_異なる文字(self):
        """異なる文字の場合距離が0より大きい"""
        distance = calculate_kana_distance("カナ", "ハナ")
        assert distance > 0.0

    def test_正常系_空文字列(self):
        """空文字列の処理"""
        distance1 = calculate_kana_distance("", "アイ")
        distance2 = calculate_kana_distance("アイ", "")
        assert distance1 > 0.0
        assert distance2 > 0.0


class TestCalculateMultipleDistances:
    def test_正常系_完全一致(self):
        """完全一致の場合両方の距離が適切"""
        distances = calculate_multiple_distances("テスト", "テスト")

        assert "cer" in distances
        assert "kana_distance" in distances
        assert distances["cer"] == 0.0
        assert distances["kana_distance"] == 0.0

    def test_正常系_異なる文字(self):
        """異なる文字の場合両方の距離が計算される"""
        distances = calculate_multiple_distances("カナ", "ハナ")

        assert "cer" in distances
        assert "kana_distance" in distances
        assert distances["cer"] > 0.0
        assert distances["kana_distance"] > 0.0


class TestEvaluateDataset:
    def test_正常系_基本処理(self):
        """基本的なデータセット評価処理"""
        # モックデータセット
        dataset = [
            {
                "id": "test_001",
                "wav_file": "test.wav",
                "wav_path": "/path/to/test.wav",
                "text": "テスト",
            }
        ]

        # モック転写関数
        def mock_transcribe(wav_path, **kwargs):
            return "テスト"

        result = evaluate_dataset(dataset, mock_transcribe)

        assert result["processed_count"] == 1
        assert result["total_count"] == 1
        assert "average_cer" in result
        assert "average_kana_distance" in result
        assert len(result["individual_results"]) == 1

        individual = result["individual_results"][0]
        assert individual["id"] == "test_001"
        assert individual["reference"] == "テスト"
        assert individual["hypothesis"] == "テスト"
        assert individual["cer"] == 0.0
        assert individual["kana_distance"] == 0.0

    def test_正常系_複数サンプル(self):
        """複数サンプルの評価処理"""
        dataset = [
            {
                "id": "test_001",
                "wav_file": "test1.wav",
                "wav_path": "/path/to/test1.wav",
                "text": "アイウ",
            },
            {
                "id": "test_002",
                "wav_file": "test2.wav",
                "wav_path": "/path/to/test2.wav",
                "text": "カキク",
            },
        ]

        def mock_transcribe(wav_path, **kwargs):
            if "test1" in wav_path:
                return "アイウ"  # 完全一致
            else:
                return "アイク"  # 部分一致

        result = evaluate_dataset(dataset, mock_transcribe)

        assert result["processed_count"] == 2
        assert result["total_count"] == 2
        assert len(result["individual_results"]) == 2

    def test_異常系_wav_pathなし(self):
        """wav_pathが無い場合のスキップ処理"""
        dataset = [
            {
                "id": "test_001",
                "wav_file": "test.wav",
                "text": "テスト",
                # wav_pathが無い
            }
        ]

        def mock_transcribe(wav_path, **kwargs):
            return "テスト"

        result = evaluate_dataset(dataset, mock_transcribe)

        assert result["processed_count"] == 0
        assert result["total_count"] == 1
        assert len(result["individual_results"]) == 0

    def test_異常系_転写エラー(self):
        """転写関数でエラーが発生した場合"""
        dataset = [
            {
                "id": "test_001",
                "wav_file": "test.wav",
                "wav_path": "/path/to/test.wav",
                "text": "テスト",
            }
        ]

        def mock_transcribe(wav_path, **kwargs):
            raise Exception("転写エラー")

        result = evaluate_dataset(dataset, mock_transcribe)

        assert result["processed_count"] == 0
        assert result["total_count"] == 1
        assert len(result["individual_results"]) == 0


class TestPrintEvaluationResults:
    def test_正常系_基本出力(self, capsys):
        """基本的な結果出力のテスト"""
        evaluation_results = {
            "processed_count": 2,
            "total_count": 2,
            "average_cer": 0.25,
            "average_kana_distance": 5.5,
            "individual_results": [
                {
                    "id": "test_001",
                    "wav_file": "test1.wav",
                    "reference": "アイウ",
                    "hypothesis": "アイク",
                    "cer": 0.33,
                    "kana_distance": 4.2,
                }
            ],
        }

        print_evaluation_results(evaluation_results, "test")

        captured = capsys.readouterr()
        assert "EVALUATION RESULTS (test)" in captured.out
        assert "CER (Character Error Rate): 0.2500" in captured.out
        assert "Kana Distance (kanasim):    5.5000" in captured.out
        assert "test_001" in captured.out
        assert "CER: 0.3300" in captured.out
        assert "Kana Distance: 4.2000" in captured.out

    def test_正常系_モード指定なし(self, capsys):
        """モード指定なしの出力テスト"""
        evaluation_results = {
            "processed_count": 1,
            "total_count": 1,
            "average_cer": 0.0,
            "average_kana_distance": 0.0,
            "individual_results": [],
        }

        print_evaluation_results(evaluation_results)

        captured = capsys.readouterr()
        assert "EVALUATION RESULTS" in captured.out
        assert "(test)" not in captured.out
