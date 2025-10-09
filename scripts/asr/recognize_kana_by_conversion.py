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
import sys

import torch

from kanagib.speech_recognition import MODEL_CONFIGS, transcribe_audio


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
            args.audio,
            mode="conversion",
            model_name=args.model,
            device=device,
            return_raw=True,
        )

        print("\n=== Results ===")
        print(f"Raw ASR Output: {raw_text}")
        print(f"Katakana (only): {katakana_only}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
