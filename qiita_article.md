# 色んな方法でカナASRしてみた

## 背景

音声認識（ASR）の結果をカタカナで取得したいニーズは意外とあります。特に日本語音声処理では、発音の表記や音韻解析において、ひらがなよりもカタカナの方が適している場面があります。

本記事では、日本語音声認識の結果をカタカナで取得するために、複数の異なるアプローチを試してみました。具体的には以下の2つの主要な手法を比較検討します：

1. **後処理でカナ変換**: 通常のASRモデルで認識した結果を後処理でカタカナに変換
2. **トークン制限**: ASRモデルのデコード時にカタカナトークンのみに制限

それぞれの手法について、通常のASRモデルとひらがなASRモデルの両方で検証し、精度や実装の容易さを比較します。

実験に使用したデータは、統計的に生成したカタカナのでたらめ文字列（gibberish）をVOICEVOXで音声合成したものです。これにより、実際の単語に依存しない純粋な音韻認識性能を評価できます。

## 後処理でカナ変換

### 通常ASR

まず最も標準的なアプローチとして、汎用的な日本語ASRモデルで認識した結果を後処理でカタカナに変換する方法を試します。

```python
#!/usr/bin/env python3
"""
音声認識してpyopenjtalkでカタカナに変換するスクリプト

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh
"""

import re
import librosa
import numpy as np
import pyopenjtalk
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

TARGET_SR = 16_000

def text_to_katakana(text: str) -> str:
    """テキストをpyopenjtalkでカタカナに変換し、カタカナのみを抽出"""
    # 読み仮名を取得
    phonemes = pyopenjtalk.g2p(text, kana=True)

    # カタカナ（ア-ヴ）と長音記号（ー）のみを抽出
    katakana_pattern = r"[ア-ヴー]+"
    katakana_matches = re.findall(katakana_pattern, phonemes)
    return "".join(katakana_matches)

def transcribe_audio(audio_path: str, model_name: str = "andrewmcdowell"):
    """音声ファイルを音声認識してpyopenjtalkでカタカナに変換"""

    # モデル設定
    MODEL_CONFIGS = {
        "andrewmcdowell": {
            "model_id": "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana",
        },
        "reazon": {
            "model_id": "reazon-research/japanese-wav2vec2-base-rs35kh",
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[model_name]

    # モデルとプロセッサーの準備
    processor = AutoProcessor.from_pretrained(config["model_id"])
    model = Wav2Vec2ForCTC.from_pretrained(config["model_id"]).to(device)

    # 音声読み込み（16kHzモノラル、前後0.5秒パディング）
    wav, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    pad = int(0.5 * TARGET_SR)
    wav = np.pad(wav, pad_width=pad)

    # 推論
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    # デコードして結果を取得
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # pyopenjtalkでカタカナに変換
    katakana_result = text_to_katakana(transcription)

    return transcription, katakana_result
```

このアプローチの特徴：

**メリット:**
- 既存の高性能ASRモデルをそのまま利用可能
- pyopenjtalkの辞書により一般的な日本語の読み方を正確に変換
- 実装が比較的シンプル

**デメリット:**
- pyopenjtalkが認識できない文字列（でたらめ文字列など）は変換されない
- 2段階処理のためASR誤りが後処理でも修正されない
- 辞書にない固有名詞等で変換精度が落ちる可能性

### ひらがなASR

次に、ひらがなに特化したASRモデルを使用して、その結果をカタカナに変換する方法です。

```python
def transcribe_with_hiragana_model(audio_path: str):
    """ひらがなASRモデルで認識後、jaconvでカタカナに変換"""
    import jaconv

    # ひらがなモデルで認識（上記と同じ流れ）
    transcription, _ = transcribe_audio(audio_path, model_name="andrewmcdowell")

    # jaconvでひらがなをカタカナに変換
    katakana_result = jaconv.hira2kata(transcription)

    return transcription, katakana_result
```

ひらがなASRモデルの場合、出力が既にかな文字に限定されているため、jaconvによる単純な文字変換で十分です。

## トークン制限

### 通常ASR

ASRモデルのデコード時に、カタカナトークンのみを許可する語彙制限を適用する方法です。

```python
def is_kana_character(char: str) -> bool:
    """ひらがな・カタカナ・長音符かどうか判定"""
    code = ord(char)
    # ひらがな (0x3041-0x3096) + カタカナ (0x30A0-0x30FF)
    return (0x3041 <= code <= 0x3096) or (0x30A0 <= code <= 0x30FF)

def create_kana_vocabulary_mask(processor: AutoProcessor, model_name: str) -> torch.Tensor:
    """モデルに応じてかな文字のみの語彙マスクを作成"""
    vocab = processor.tokenizer.get_vocab()
    vocab_size = len(vocab)
    allowed_mask = torch.ones(vocab_size, dtype=torch.bool)

    for token, token_id in vocab.items():
        # 特殊トークンは保持
        if token.startswith("<") and token.endswith(">"):
            continue
        if token in ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "|"]:
            continue

        # 句読点を除外
        if token in ["、", "。", "，", "．", ",", ".", "!", "?", "！", "？", "・"]:
            allowed_mask[token_id] = False
            continue

        # モデルごとのトークン処理
        check_token = token.lstrip("▁") if model_name == "reazon" else token

        # トークンが全てかな文字でない場合は除外
        if (check_token and
            not all(is_kana_character(c) or c in [" ", "　"]
                   for c in check_token if c.strip()) and
            check_token.strip()):
            allowed_mask[token_id] = False

    return allowed_mask

def transcribe_with_kana_restriction(audio_path: str, model_name: str = "andrewmcdowell"):
    """カナ文字制限付きでASRを実行"""
    import jaconv

    # モデルとプロセッサーの準備（前述と同じ）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... モデル読み込み処理 ...

    # カナ文字語彙マスク作成
    kana_mask = create_kana_vocabulary_mask(processor, model_name)

    # 音声読み込み
    # ... 音声処理 ...

    # 推論（語彙制限適用）
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

        # カナ文字以外のトークンを大幅に抑制
        logits[:, :, ~kana_mask] = logits[:, :, ~kana_mask] - 1000

        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.decode(predicted_ids[0])

    # ひらがなをカタカナに変換
    katakana_result = jaconv.hira2kata(transcription)

    return transcription, katakana_result
```

### ひらがなASR

ひらがなASRモデルに対して同様の語彙制限を適用します。このケースでは、元々カナ文字に特化しているため、より制限が効果的に働きます。

```python
def transcribe_hiragana_with_restriction(audio_path: str):
    """ひらがなASRモデルにカナ制限を適用"""
    return transcribe_with_kana_restriction(audio_path, model_name="andrewmcdowell")
```

## 実験結果

統計的に生成したカタカナのでたらめ文字列（例：「ノコーカンオブッシュツカイルダーヤキモロニセキホーレテオールラカア」）をVOICEVOXで音声合成し、各手法で認識精度を比較しました。

### サンプルデータ
```json
{
  "id": "10a97e2e-4ea7-492c-8a27-0667a5c52138",
  "text": "ノコーカンオブッシュツカイルダーヤキモロニセキホーレテオールラカア",
  "wav_file": "gibberish_10a97e2e-4ea7-492c-8a27-0667a5c52138_ノコーカンオブッシュツカイルダーヤキモロニセキホーレテオールラカア.wav",
  "wav_speaker": 3,
  "wav_duration": 4.512
}
```

### 手法別の特徴比較

| 手法 | メリット | デメリット | 適用場面 |
|------|----------|------------|----------|
| 通常ASR + 後処理 | 高精度なASRモデル活用、実装簡単 | 辞書依存、でたらめ文字列に弱い | 一般的な日本語音声 |
| ひらがなASR + 後処理 | かな文字特化、文字変換が確実 | モデル選択肢が限定的 | かな表記が重要な用途 |
| 通常ASR + トークン制限 | でたらめ文字列対応、リアルタイム | 実装複雑、一部精度低下 | 固有名詞や造語を含む音声 |
| ひらがなASR + トークン制限 | 最も制限が効果的 | オーバーフィッティングリスク | 純粋な音韻認識が必要 |

## おわりに

カタカナASRには複数のアプローチがあり、それぞれに特徴があります：

1. **後処理でのカナ変換**は実装が容易で、一般的な日本語には有効ですが、でたらめ文字列や固有名詞には限界があります。

2. **トークン制限**は辞書に依存せず、任意の音韻組み合わせに対応できますが、実装がやや複雑になります。

使用する場面に応じて適切な手法を選択することが重要です。一般的な日本語音声処理には後処理アプローチ、音韻研究や固有名詞を多く含む音声にはトークン制限アプローチが適しているでしょう。

今回の実験コードは[GitHub](https://github.com/jiroshimaya/kana-gibberish)で公開していますので、ご参考ください。

### 参考リンク

- [AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana](https://huggingface.co/AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana)
- [reazon-research/japanese-wav2vec2-base-rs35kh](https://huggingface.co/reazon-research/japanese-wav2vec2-base-rs35kh)
- [VOICEVOX](https://voicevox.hiroshiba.jp/)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [jaconv](https://github.com/ikegami-yukino/jaconv)
