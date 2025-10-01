# 統合音声認識評価ツール

`scripts/eval_recognition.py` は、JSON設定ファイルから複数の評価条件を読み込んで一括実行する統合評価ツールです。

## 特徴

- **複数条件一括実行**: 異なるモデル・データセット・認識モードを設定ファイルで管理
- **GPU自動最適化**: GPU環境では自動的にバッチ処理を適用（filter方式のみ）
- **JSON設定ファイル**: シンプルで分かりやすいJSON形式の設定ファイル
- **結果保存**: 評価結果をJSON形式で保存可能
- **ドライラン**: 実行前の設定確認機能

## 使用方法

### 基本的な使用方法

```bash
# JSON設定ファイルで実行
uv run python scripts/eval_recognition.py config/evaluation_example.json

# テスト用設定で実行
uv run python scripts/eval_recognition.py config/evaluation_simple.json

# 結果をファイルに保存
uv run python scripts/eval_recognition.py config/evaluation_example.json --output results.json

# 設定確認のみ（実際の評価は実行しない）
uv run python scripts/eval_recognition.py config/evaluation_example.json --dry-run
```

## 設定ファイル形式

設定ファイルはJSON形式で作成します：

```json
{
  "name": "評価バッチ名",
  "description": "評価の説明",
  "configs": [
    {
      "name": "評価設定名",
      "description": "この評価の説明",
      "json_file": "datasets/sample/gibberish_with_wav.json",
      "model": "andrewmcdowell",
      "mode": "filter",
      "batch_size": 8,
      "wav_dir": null,
      "use_batch": null
    }
  ]
}
```

## 設定項目

### バッチ設定

- `name`: 評価バッチの名前
- `description`: 評価バッチの説明

### 個別評価設定

- `name` (必須): 評価設定の名前
- `json_file` (必須): JSONデータセットファイルパス
- `model` (必須): 使用するモデル (`andrewmcdowell` または `reazon`)
- `mode` (必須): 音声認識モード (`filter` または `conversion`)
- `description`: 評価の説明（オプション）
- `wav_dir`: 音声ファイルディレクトリ（オプション、デフォルトはJSONファイルと同じ場所のwavフォルダ）
- `batch_size`: バッチサイズ（オプション、デフォルト4）
- `use_batch`: バッチ処理強制指定（オプション）
  - `null` (デフォルト): 自動判定（GPU環境 + filter方式でバッチ処理）
  - `true`: 強制的にバッチ処理使用
  - `false`: 強制的に単体処理使用

## バッチ処理について

- **自動判定**: GPU環境でfilter方式の場合、自動的にバッチ処理を使用
- **conversion方式**: 現在バッチ処理には対応していないため、常に単体処理
- **CPU環境**: メモリ効率を考慮して単体処理を使用

## 出力例

```
===============================================================================
Starting evaluation: sample_test_filter
Model: andrewmcdowell
Mode: filter
JSON: datasets/sample/gibberish_with_wav.json
Description: サンプルデータでのFilter方式テスト
===============================================================================
Device: cuda
Using batch processing (batch_size=2)
Loaded 10 samples from dataset

Evaluation Results (filter (batch, size=2)):
  Total samples: 10
  Average CER: 0.1250
  Total errors: 15
  Total characters: 120

Execution completed:
  Total time: 5.42 seconds (0.09 minutes)
  Average time per sample: 0.54 seconds
  Device: cuda
  Batch size: 2

================================================================================
EVALUATION SUMMARY
================================================================================
Total evaluations: 2
Total samples processed: 20
Total execution time: 12.34 seconds (0.21 minutes)
Average time per evaluation: 6.17 seconds

Detailed results:
  sample_test_filter: CER=0.1250, samples=10, time=5.4s, model=andrewmcdowell, mode=filter (batch=2)
  sample_test_conversion: CER=0.0980, samples=10, time=6.9s, model=andrewmcdowell, mode=conversion
================================================================================
```

## 設定ファイル例

### テスト用（簡単）
`config/evaluation_simple.json` - 少数のサンプルデータでのテスト用

### 本格評価用
`config/evaluation_example.json` - 複数データセット・モデルでの本格評価用

## 注意事項

- 設定ファイル内のパスは、スクリプト実行ディレクトリからの相対パスまたは絶対パス
- GPU環境では自動的にバッチ処理が適用されるため、メモリ使用量に注意
- 大量のデータセットを評価する場合は、十分な実行時間を確保してください
