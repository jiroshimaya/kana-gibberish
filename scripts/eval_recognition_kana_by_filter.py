#!/usr/bin/env python3
"""
Kana ASR Evaluation Tool for JSON datasets.

Evaluates speech recognition models on JSON datasets containing audio files
and reference texts, calculating Character Error Rate (CER) for each sample.

Supported models:
  - AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
  - reazon-research/japanese-wav2vec2-base-rs35kh

Usage:
  python eval_recognition_kana_by_filter.py dataset.json
  python eval_recognition_kana_by_filter.py dataset.json --model reazon
  python eval_recognition_kana_by_filter.py dataset.json --wav-dir /path/to/wav/files
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import editdistance
import jaconv
import librosa
import numpy as np
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
        check_token = token.lstrip("▁") if model_name == "reazon" else token

        # トークンが全てかな文字でない場合は除外
        if (
            check_token
            and not all(
                is_kana_character(c) or c in [" ", "　"]
                for c in check_token
                if c.strip()
            )
            and check_token.strip()
        ):
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
    pad_sec = 0.5  # 0.5秒パディング固定

    # モデルとプロセッサーの準備
    model, processor = setup_model_and_processor(model_name, device)

    # 語彙マスクの作成
    vocab_mask = create_kana_vocabulary_mask(processor, model_name)
    vocab_mask = vocab_mask.to(device)

    allowed_count = vocab_mask.sum().item()
    total_count = len(vocab_mask)
    print(
        f"[INFO] Vocabulary restricted to kana characters. "
        f"Allowed tokens: {allowed_count}/{total_count}"
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


def load_json_dataset(json_path: str, wav_dir: str = None) -> List[Dict[str, str]]:
    """
    JSONファイルからデータセットを読み込み
    
    Args:
        json_path: JSONファイルのパス
        wav_dir: 音声ファイルのディレクトリ（Noneの場合はJSONファイルと同じディレクトリのwavフォルダを使用）
    
    Returns:
        データセットのリスト
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # wav_dirが指定されていない場合のデフォルト設定
    if wav_dir is None:
        json_parent = Path(json_path).parent
        wav_dir = json_parent / "wav"
    
    wav_dir = Path(wav_dir)
    
    # 音声ファイルのパスを絶対パスに変換
    for item in data:
        wav_file = item['wav_file']
        wav_path = wav_dir / wav_file
        
        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}")
            continue
            
        item['wav_path'] = str(wav_path)
    
    return data


def evaluate_dataset(
    dataset: List[Dict[str, str]], 
    model_name: str = "andrewmcdowell", 
    device: torch.device = None
) -> Dict[str, float]:
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
        if 'wav_path' not in item:
            logger.warning(f"Skipping item {i}: no wav_path")
            continue
            
        wav_path = item['wav_path']
        reference = item['text']
        item_id = item.get('id', f'item_{i}')
        
        try:
            logger.info(f"Processing {i+1}/{len(dataset)}: {item_id}")
            
            # 音声認識実行
            hypothesis = transcribe_audio(wav_path, model_name=model_name, device=device)
            
            # CER計算
            cer = calculate_cer(reference, hypothesis)
            
            result = {
                'id': item_id,
                'wav_file': item['wav_file'],
                'reference': reference,
                'hypothesis': hypothesis,
                'cer': cer
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
        'individual_results': results,
        'average_cer': average_cer,
        'processed_count': processed_count,
        'total_count': len(dataset)
    }


def main():
    ap = argparse.ArgumentParser(description="Kana ASR Evaluation Tool for JSON datasets")
    ap.add_argument(
        "json_file", 
        help="JSONデータセットファイル（gibberish_with_wav.json形式）"
    )
    ap.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="andrewmcdowell",
        help="使用するモデル (default: andrewmcdowell)",
    )
    ap.add_argument(
        "--wav-dir",
        help="音声ファイルディレクトリ（デフォルト: JSONファイルと同じディレクトリのwavフォルダ）"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, model={args.model}")
    
    json_path = Path(args.json_file)
    
    # JSONファイル存在確認
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    
    if json_path.suffix.lower() != '.json':
        logger.error(f"Input file must be a JSON file, got: {json_path.suffix}")
        sys.exit(1)
    
    try:
        logger.info("JSON dataset evaluation mode")
        
        # データセット読み込み
        dataset = load_json_dataset(str(json_path), args.wav_dir)
        
        if not dataset:
            logger.error("No valid data found in JSON file")
            sys.exit(1)
        
        # バッチ評価実行
        evaluation_results = evaluate_dataset(dataset, model_name=args.model, device=device)
        
        # 結果出力
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nProcessed: {evaluation_results['processed_count']}/{evaluation_results['total_count']} samples")
        print(f"Average CER: {evaluation_results['average_cer']:.4f}")
        
        print("\n" + "-"*50)
        print("Individual Results:")
        print("-"*50)
        
        for result in evaluation_results['individual_results']:
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
