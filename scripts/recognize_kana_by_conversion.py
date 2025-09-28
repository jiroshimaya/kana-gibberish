"""
音声認識してpyopenjtalkでカタカナに変換するスクリプト

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Usage:
  python recognize_kana_by_conversion.py path/to/audio.wav
  python recognize_kana_by_conversion.py path/to/audio.wav --model reazon
"""

import argparse
import re
import sys

import librosa
import numpy as np
import pyopenjtalk
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


def text_to_katakana(text: str) -> str:
    """テキストをpyopenjtalkでカタカナに変換し、カタカナのみを抽出"""
    # 読み仮名を取得
    phonemes = pyopenjtalk.g2p(text, kana=True)

    # カタカナ（ア-ヴ）と長音記号（ー）のみを抽出
    katakana_pattern = r"[ア-ヴー]+"
    katakana_matches = re.findall(katakana_pattern, phonemes)
    return "".join(katakana_matches)


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
) -> tuple[str, str]:
    """音声ファイルを音声認識してpyopenjtalkでカタカナに変換"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 決め打ち設定
    pad_sec = 0.5  # 0.5秒パディング固定

    # モデルとプロセッサーの準備
    model, processor = setup_model_and_processor(model_name, device)

    print("[INFO] Using ASR without vocabulary restriction")

    # 音声読み込み
    wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)

    # 入力準備
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")

    # デバイスに移動
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    # 推論（語彙制限なし）
    with torch.inference_mode():
        logits = model(**inputs).logits  # [B, T, V]
        pred_ids = torch.argmax(logits.detach().float().cpu(), dim=-1)[0]

    raw_text = processor.decode(pred_ids, skip_special_tokens=True)

    # pyopenjtalkでカタカナに変換（カタカナのみ抽出済み）
    katakana_only = text_to_katakana(raw_text)

    return raw_text, katakana_only


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
        raw_text, katakana_only = transcribe_audio(
            args.audio, model_name=args.model, device=device
        )

        print("\n=== Results ===")
        print(f"Raw ASR Output: {raw_text}")
        print(f"Katakana (only): {katakana_only}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
