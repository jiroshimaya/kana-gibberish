#!/usr/bin/env python3
"""
Unified Kana-first ASR supporting multiple models.

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Settings (fixed):
  - dtype: float32 (for stability)
  - padding: 0.5sec before/after audio

Usage:
  python recognize_kana.py path/to/audio.wav
  python recognize_kana.py path/to/audio.wav --model reazon
"""

import argparse
import sys

import jaconv
import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# モデル設定
MODEL_CONFIGS = {
    "andrewmcdowell": {
        "model_id": "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana",
    },
    "reazon": {
        "model_id": "reazon-research/japanese-wav2vec2-base-rs35kh",
    },
}

TARGET_SR = 16_000


def is_kana_character(char: str) -> bool:
    """ひらがな・カタカナ・長音符かどうか判定"""
    code = ord(char)
    # ひらがな (0x3041-0x3096) + カタカナ (0x30A0-0x30FF)
    return (0x3041 <= code <= 0x3096) or (0x30A0 <= code <= 0x30FF)


def create_kana_vocabulary_mask(
    processor: AutoProcessor, model_name: str
) -> torch.Tensor:
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

        # 読点・句読点を除外
        if token in ["、", "。", "，", "．", ",", ".", "!", "?", "！", "？", "・"]:
            allowed_mask[token_id] = False
            continue

        # モデルごとのトークン処理
        if model_name == "reazon":
            # Reazonモデル: 先頭のSentencePieceマーカー "▁" を除去
            check_token = token.lstrip("▁")
        else:
            # AndrewMcDowellモデル: そのまま
            check_token = token

        # トークンが全てかな文字でない場合は除外
        if check_token and not all(
            is_kana_character(c) or c in [" ", "　"] for c in check_token if c.strip()
        ):
            if check_token.strip():  # 空白のみのトークンは許可
                allowed_mask[token_id] = False

    return allowed_mask


def load_audio_mono_16k(path: str, pad_sec: float) -> np.ndarray:
    """音声ファイルを16kHzモノラルで読み込み、オプションでパディング追加"""
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if pad_sec and pad_sec > 0:
        pad = int(pad_sec * TARGET_SR)
        wav = np.pad(wav, pad_width=pad)
    # クリッピング対策
    if np.max(np.abs(wav)) > 1.0:
        wav = wav / (np.max(np.abs(wav)) + 1e-8)
    return wav.astype(np.float32)


def setup_model_and_processor(
    model_name: str, device: torch.device
) -> tuple[Wav2Vec2ForCTC, AutoProcessor]:
    """モデルとプロセッサーをセットアップ"""
    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]

    print(f"[INFO] Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # デバイス設定（float32固定）
    model = model.to(device)

    return model, processor


def transcribe_audio(
    audio_path: str,
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
) -> str:
    """音声ファイルをかな文字で転写"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 決め打ち設定
    dtype = torch.float32  # 安定性重視でfloat32固定
    pad_sec = 0.5  # 0.5秒パディング固定

    # モデルとプロセッサーの準備
    model, processor = setup_model_and_processor(model_name, device)

    # 語彙マスクの作成
    vocab_mask = create_kana_vocabulary_mask(processor, model_name)
    vocab_mask = vocab_mask.to(device)

    print(
        f"[INFO] Vocabulary restricted to kana characters. Allowed tokens: {vocab_mask.sum().item()}/{len(vocab_mask)}"
    )

    # 音声読み込み
    wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)

    # 入力準備
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")

    # デバイスに移動
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    # 推論
    with torch.inference_mode():
        logits = model(**inputs).logits  # [B, T, V]

        # 語彙制限を適用
        logits = logits.masked_fill(
            ~vocab_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        pred_ids = torch.argmax(logits.detach().float().cpu(), dim=-1)[0]

    text = processor.decode(pred_ids, skip_special_tokens=True)

    # カタカナ統一
    text = jaconv.hira2kata(text)

    return text


def main():
    ap = argparse.ArgumentParser(description="Unified Kana-first ASR")
    ap.add_argument("audio", help="入力音声ファイル（wav/その他librosa対応）")
    ap.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="andrewmcdowell",
        help="使用するモデル (default: andrewmcdowell)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, model={args.model}")

    # 転写実行
    try:
        text = transcribe_audio(args.audio, model_name=args.model, device=device)

        print("\n=== Transcription ===")
        print(text)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
