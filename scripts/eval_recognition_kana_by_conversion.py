"""
Kana ASR Evaluation Tool using pyopenjtalk conversion for JSON datasets.

Evaluates speech recognition models on JSON datasets, converting ASR output
to kata    with Path(json_path).open(encoding='utf-8') as f:ana using pyopenjtalk and calculating Character Error Rate (CER).

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Usage:
  python eval_recognition_kana_by_conversion.py dataset.json
  python eval_recognition_kana_by_conversion.py dataset.json --model reazon
  python eval_recognition_kana_by_conversion.py dataset.json \
    --wav-dir /path/to/wav/files
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import editdistance
import librosa
import numpy as np
import pyopenjtalk
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
) -> str:
    """音声ファイルを音声認識してpyopenjtalkでカタカナに変換（カタカナのみ返す）"""

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

    return katakana_only


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER)を計算

    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト

    Returns:
        CER値 (0.0-1.0)
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    # 編集距離を計算
    edit_dist = editdistance.eval(reference, hypothesis)
    cer = edit_dist / len(reference)
    return cer


def load_json_dataset(
    json_path: str, wav_dir: str | None = None
) -> list[dict[str, str]]:
    """
    JSONファイルからデータセットを読み込み

    Args:
        json_path: JSONファイルのパス
        wav_dir: 音声ファイルのディレクトリ
            （Noneの場合はJSONファイルと同じディレクトリのwavフォルダを使用）

    Returns:
        データセットのリスト
    """
    with Path(json_path).open(encoding="utf-8") as f:
        data = json.load(f)

    # wav_dirが指定されていない場合のデフォルト設定
    if wav_dir is None:
        json_parent = Path(json_path).parent
        wav_dir = json_parent / "wav"

    wav_dir = Path(wav_dir)

    # 音声ファイルのパスを絶対パスに変換
    for item in data:
        wav_file = item["wav_file"]
        wav_path = wav_dir / wav_file

        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}")
            continue

        item["wav_path"] = str(wav_path)

    return data


def evaluate_dataset(
    dataset: list[dict[str, str]],
    model_name: str = "andrewmcdowell",
    device: torch.device = None,
) -> dict[str, float]:
    """
    データセット全体の音声認識評価を実行

    Args:
        dataset: 評価データセット
        model_name: 使用するモデル名
        device: 使用するデバイス

    Returns:
        評価結果の辞書（individual_results, average_cer等）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting evaluation with model: {model_name}, device: {device}")
    logger.info(f"Total samples: {len(dataset)}")

    results = []
    total_cer = 0.0
    processed_count = 0

    for i, item in enumerate(dataset):
        if "wav_path" not in item:
            logger.warning(f"Skipping item {i}: no wav_path")
            continue

        wav_path = item["wav_path"]
        reference = item["text"]
        item_id = item.get("id", f"item_{i}")

        try:
            logger.info(f"Processing {i+1}/{len(dataset)}: {item_id}")

            # 音声認識実行（pyopenjtalk変換あり）
            hypothesis = transcribe_audio(
                wav_path, model_name=model_name, device=device
            )

            # CER計算
            cer = calculate_cer(reference, hypothesis)

            result = {
                "id": item_id,
                "wav_file": item["wav_file"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": cer,
            }

            results.append(result)
            total_cer += cer
            processed_count += 1

            logger.info(f"  CER: {cer:.4f}")
            logger.info(f"  REF: {reference}")
            logger.info(f"  HYP: {hypothesis}")

        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            continue

    # 結果まとめ
    average_cer = total_cer / processed_count if processed_count > 0 else 0.0

    return {
        "individual_results": results,
        "average_cer": average_cer,
        "processed_count": processed_count,
        "total_count": len(dataset),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Kana ASR Evaluation Tool using pyopenjtalk conversion"
    )
    ap.add_argument(
        "json_file", help="JSONデータセットファイル（gibberish_with_wav.json形式）"
    )
    ap.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="andrewmcdowell",
        help="使用するモデル (default: andrewmcdowell)",
    )
    ap.add_argument(
        "--wav-dir",
        help="音声ファイルディレクトリ\n"
        "（デフォルト: JSONファイルと同じディレクトリのwavフォルダ）",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, model={args.model}")

    json_path = Path(args.json_file)

    # JSONファイル存在確認
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    if json_path.suffix.lower() != ".json":
        logger.error(f"Input file must be a JSON file, got: {json_path.suffix}")
        sys.exit(1)

    try:
        logger.info("JSON dataset evaluation mode (with pyopenjtalk conversion)")

        # データセット読み込み
        dataset = load_json_dataset(str(json_path), args.wav_dir)

        if not dataset:
            logger.error("No valid data found in JSON file")
            sys.exit(1)

        # バッチ評価実行
        evaluation_results = evaluate_dataset(
            dataset, model_name=args.model, device=device
        )

        # 結果出力
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS (pyopenjtalk conversion)")
        print("=" * 50)

        processed = evaluation_results["processed_count"]
        total = evaluation_results["total_count"]
        print(f"\nProcessed: {processed}/{total} samples")
        print(f"Average CER: {evaluation_results['average_cer']:.4f}")

        print("\n" + "-" * 50)
        print("Individual Results:")
        print("-" * 50)

        for result in evaluation_results["individual_results"]:
            print(f"\nID: {result['id']}")
            print(f"File: {result['wav_file']}")
            print(f"CER: {result['cer']:.4f}")
            print(f"REF: {result['reference']}")
            print(f"HYP: {result['hypothesis']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
