"""
音声認識の共通処理

このモジュールには音声認識に関する共通的な機能を含む：
- モデル設定
- 音声ファイルの読み込み
- モデルのセットアップ
- 音声認識処理（フィルター方式・変換方式）
- テキスト変換処理
"""

import logging
import re
from typing import Literal

import jaconv
import librosa
import numpy as np
import pyopenjtalk
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

logger = logging.getLogger(__name__)

# モデル設定（公式サンプルに基づく最適化設定）
MODEL_CONFIGS = {
    "andrewmcdowell": {
        "model_id": "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana",
        "use_flash_attention": True,  # 公式サンプルに合わせて有効化
        "torch_dtype": torch.bfloat16,  # 公式サンプルに合わせてbf16使用
    },
    "reazon": {
        "model_id": "reazon-research/japanese-wav2vec2-base-rs35kh",
        "use_flash_attention": True,  # 公式サンプルに合わせて有効化
        "torch_dtype": torch.bfloat16,  # 公式サンプルに合わせてbf16使用
    },
}

TARGET_SR = 16_000


def is_kana_character(char: str) -> bool:
    """ひらがな・カタカナ・長音符かどうか判定

    Args:
        char: 判定する文字

    Returns:
        かな文字かどうか
    """
    code = ord(char)
    # ひらがな (0x3041-0x3096) + カタカナ (0x30A0-0x30FF)
    return (0x3041 <= code <= 0x3096) or (0x30A0 <= code <= 0x30FF)


def text_to_katakana(text: str) -> str:
    """テキストをpyopenjtalkでカタカナに変換し、カタカナのみを抽出

    Args:
        text: 変換対象のテキスト

    Returns:
        カタカナのみのテキスト
    """
    # 読み仮名を取得
    phonemes = pyopenjtalk.g2p(text, kana=True)
    # ヲをオに変換
    phonemes = phonemes.replace("ヲ", "オ")

    # phonemesがstrであることを保証
    if isinstance(phonemes, list):
        phonemes = "".join(phonemes)

    # カタカナ（ア-ヴ）と長音記号（ー）のみを抽出
    katakana_pattern = r"[ア-ヴー]+"
    katakana_matches = re.findall(katakana_pattern, phonemes)
    return "".join(katakana_matches)


def create_kana_vocabulary_mask(
    processor: AutoProcessor, model_name: str
) -> torch.Tensor:
    """モデルに応じてかな文字のみの語彙マスクを作成

    Args:
        processor: Transformersのプロセッサー
        model_name: モデル名

    Returns:
        語彙マスク（かな文字のみTrueのテンソル）
    """
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


def load_audio_mono_16k(path: str, pad_sec: float = 0.5) -> np.ndarray:
    """音声ファイルを16kHzモノラルで読み込み、オプションでパディング追加

    Args:
        path: 音声ファイルのパス
        pad_sec: パディング秒数

    Returns:
        音声波形データ
    """
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
    """モデルとプロセッサーをセットアップ

    Args:
        model_name: 使用するモデル名
        device: 使用するデバイス

    Returns:
        モデルとプロセッサーのタプル
    """
    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]

    logger.info(f"Loading model: {model_id}")

    # FlashAttention設定の確認
    use_flash_attention = config.get("use_flash_attention", False)
    torch_dtype = config.get("torch_dtype", torch.float32)

    processor = AutoProcessor.from_pretrained(model_id)

    # GPU使用時のみFlashAttentionと最適化を適用
    model_kwargs = {}
    if device and device.type == "cuda":
        if use_flash_attention:
            try:
                # FlashAttentionが利用可能か確認
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("FlashAttention 2.0 enabled")
            except Exception as e:
                logger.warning(f"FlashAttention not available, using default: {e}")

        model_kwargs["dtype"] = torch_dtype
        logger.info(f"Using dtype: {torch_dtype}")

    model = Wav2Vec2ForCTC.from_pretrained(model_id, **model_kwargs)

    # デバイス設定
    model = model.to(device)  # type:ignore

    # GPU使用時はメモリ使用量を確認
    if device and device.type == "cuda":
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"GPU memory allocated: {allocated_memory:.2f} GB")

    return model, processor


def transcribe_audio_with_filter(
    audio_path: str,
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
) -> str:
    """音声ファイルをかな文字フィルターで転写

    Args:
        audio_path: 音声ファイルのパス
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数

    Returns:
        認識結果のカタカナテキスト
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルとプロセッサーの準備
    model, processor = setup_model_and_processor(model_name, device)

    # 語彙マスクの作成
    vocab_mask = create_kana_vocabulary_mask(processor, model_name)
    vocab_mask = vocab_mask.to(device)

    allowed_count = vocab_mask.sum().item()
    total_count = len(vocab_mask)
    logger.info(
        f"Vocabulary restricted to kana characters. "
        f"Allowed tokens: {allowed_count}/{total_count}"
    )

    # 音声読み込み
    wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)

    # 入力準備（公式サンプルに合わせた方式）
    input_values = processor(
        wav, sampling_rate=TARGET_SR, return_tensors="pt"
    ).input_values

    # GPUに移動してモデルの型に変換
    model_dtype = next(model.parameters()).dtype
    input_values = input_values.to(device).to(model_dtype)

    # 推論（公式サンプルに合わせて修正）
    with torch.inference_mode():
        logits = model(input_values).logits.cpu()  # 公式サンプル通り

        # 語彙制限を適用（vocab_maskもCPUに移動）
        vocab_mask_cpu = vocab_mask.cpu()
        logits = logits.masked_fill(
            ~vocab_mask_cpu.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        pred_ids = torch.argmax(logits, dim=-1)[0]

    text = processor.decode(pred_ids, skip_special_tokens=True)

    # カタカナ統一
    text = jaconv.hira2kata(text)

    return text


def transcribe_audio_with_conversion(
    audio_path: str,
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
    return_raw: bool = False,
) -> str | tuple[str, str]:
    """音声ファイルを音声認識してpyopenjtalkでカタカナに変換

    Args:
        audio_path: 音声ファイルのパス
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数
        return_raw: 生のASR出力も返すかどうか

    Returns:
        return_rawがTrueの場合は(生ASR出力, カタカナ)のタプル、
        Falseの場合はカタカナのみ
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルとプロセッサーの準備
    model, processor = setup_model_and_processor(model_name, device)

    logger.info("Using ASR without vocabulary restriction")

    # 音声読み込み
    wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)

    # 入力準備（公式サンプルに合わせた方式）
    input_values = processor(
        wav, sampling_rate=TARGET_SR, return_tensors="pt"
    ).input_values

    # GPUに移動してモデルの型に変換
    model_dtype = next(model.parameters()).dtype
    input_values = input_values.to(device).to(model_dtype)

    # 推論（語彙制限なし、公式サンプルに合わせて修正）
    with torch.inference_mode():
        logits = model(input_values).logits.cpu()  # 公式サンプル通り
        pred_ids = torch.argmax(logits, dim=-1)[0]

    raw_text = processor.decode(pred_ids, skip_special_tokens=True)

    # pyopenjtalkでカタカナに変換（カタカナのみ抽出済み）
    katakana_only = text_to_katakana(raw_text)

    if return_raw:
        return raw_text, katakana_only
    else:
        return katakana_only


def transcribe_audio(
    audio_path: str,
    mode: Literal["filter", "conversion"] = "filter",
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
    return_raw: bool = False,
) -> str | tuple[str, str]:
    """音声ファイルを指定された方式で転写

    Args:
        audio_path: 音声ファイルのパス
        mode: 転写方式（"filter" または "conversion"）
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数
        return_raw: 生のASR出力も返すかどうか（conversionモードのみ）

    Returns:
        認識結果のテキスト、またはタプル
    """
    if mode == "filter":
        if return_raw:
            raise ValueError("return_raw is not supported in filter mode")
        return transcribe_audio_with_filter(audio_path, model_name, device, pad_sec)
    elif mode == "conversion":
        return transcribe_audio_with_conversion(
            audio_path, model_name, device, pad_sec, return_raw
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'filter' or 'conversion'")


def transcribe_audio_batch_with_filter(
    audio_paths: list[str],
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
    batch_size: int = 4,
) -> list[str]:
    """音声ファイルのバッチをかな文字フィルターで転写

    Args:
        audio_paths: 音声ファイルのパスリスト
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数
        batch_size: バッチサイズ

    Returns:
        認識結果のカタカナテキストリスト
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Batch transcription with {model_name} on {device}")
    logger.info(f"Processing {len(audio_paths)} files with batch_size={batch_size}")

    # モデルとプロセッサーを一度だけロード
    model, processor = setup_model_and_processor(model_name, device)

    # 語彙制限マスクを作成
    vocab_mask = create_kana_vocabulary_mask(processor, model_name).to(device)
    allowed_count = vocab_mask.sum().item()
    total_count = len(vocab_mask)
    logger.info(
        f"Vocabulary restricted to kana characters. "
        f"Allowed tokens: {allowed_count}/{total_count}"
    )

    results = []

    # バッチごとに処理
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        current_batch = i // batch_size + 1
        total_batches = (len(audio_paths) + batch_size - 1) // batch_size
        logger.info(f"Processing batch {current_batch}/{total_batches}")

        # バッチの音声を読み込み
        batch_wavs = []
        for audio_path in batch_paths:
            wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)
            batch_wavs.append(wav)

        # 最大長に合わせてパディング
        max_len = max(len(wav) for wav in batch_wavs)
        padded_wavs = []
        for wav in batch_wavs:
            if len(wav) < max_len:
                padded_wav = np.pad(wav, (0, max_len - len(wav)))
            else:
                padded_wav = wav
            padded_wavs.append(padded_wav)

        # バッチテンソルを作成
        batch_array = np.stack(padded_wavs)

        # 入力準備（公式サンプルに合わせた方式）
        input_values = processor(
            batch_array, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_values

        # GPUに移動してモデルの型に変換
        model_dtype = next(model.parameters()).dtype
        input_values = input_values.to(device).to(model_dtype)

        # バッチ推論
        with torch.inference_mode():
            logits = model(input_values).logits.cpu()  # [B, T, V]

            # 語彙制限を適用（vocab_maskもCPUに移動）
            vocab_mask_cpu = vocab_mask.cpu()
            logits = logits.masked_fill(
                ~vocab_mask_cpu.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            pred_ids = torch.argmax(logits, dim=-1)  # [B, T]

        # バッチの各結果をデコード
        for j, pred_id in enumerate(pred_ids):
            text = processor.decode(pred_id, skip_special_tokens=True)
            # カタカナ統一
            katakana_text = jaconv.hira2kata(text)
            results.append(katakana_text)
            logger.info(f"  Sample {i+j+1}: {katakana_text}")

    return results


def transcribe_audio_batch_with_conversion(
    audio_paths: list[str],
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
    batch_size: int = 4,
    return_raw: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """音声ファイルのバッチをpyopenjtalk変換で転写

    Args:
        audio_paths: 音声ファイルのパスリスト
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数
        batch_size: バッチサイズ
        return_raw: 生のASR出力も返すかどうか

    Returns:
        認識結果のカタカナテキストリスト、またはタプル（生テキスト, カタカナテキスト）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Batch transcription with {model_name} on {device}")
    logger.info(f"Processing {len(audio_paths)} files with batch_size={batch_size}")

    # モデルとプロセッサーを一度だけロード
    model, processor = setup_model_and_processor(model_name, device)
    logger.info("Using ASR without vocabulary restriction")

    raw_results = []
    katakana_results = []

    # バッチごとに処理
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        current_batch = i // batch_size + 1
        total_batches = (len(audio_paths) + batch_size - 1) // batch_size
        logger.info(f"Processing batch {current_batch}/{total_batches}")

        # バッチの音声を読み込み
        batch_wavs = []
        for audio_path in batch_paths:
            wav = load_audio_mono_16k(audio_path, pad_sec=pad_sec)
            batch_wavs.append(wav)

        # 最大長に合わせてパディング
        max_len = max(len(wav) for wav in batch_wavs)
        padded_wavs = []
        for wav in batch_wavs:
            if len(wav) < max_len:
                padded_wav = np.pad(wav, (0, max_len - len(wav)))
            else:
                padded_wav = wav
            padded_wavs.append(padded_wav)

        # バッチテンソルを作成
        batch_array = np.stack(padded_wavs)

        # 入力準備（公式サンプルに合わせた方式）
        input_values = processor(
            batch_array, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_values

        # GPUに移動してモデルの型に変換
        model_dtype = next(model.parameters()).dtype
        input_values = input_values.to(device).to(model_dtype)

        # バッチ推論（語彙制限なし）
        with torch.inference_mode():
            logits = model(input_values).logits.cpu()  # [B, T, V]
            pred_ids = torch.argmax(logits, dim=-1)  # [B, T]

        # バッチの各結果をデコードしてカタカナ変換
        for j, pred_id in enumerate(pred_ids):
            raw_text = processor.decode(pred_id, skip_special_tokens=True)
            katakana_text = text_to_katakana(raw_text)

            raw_results.append(raw_text)
            katakana_results.append(katakana_text)
            logger.info(f"  Sample {i+j+1}: {katakana_text}")

    if return_raw:
        return raw_results, katakana_results
    else:
        return katakana_results
