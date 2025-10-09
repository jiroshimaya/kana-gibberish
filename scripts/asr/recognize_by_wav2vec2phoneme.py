# pip install -U torch torchaudio transformers librosa unidecode
# モデルは16kHz想定。手元のWAVを16kHz, monoにしておくと安定します。

import argparse
import collections
import collections.abc
import logging
import sys
from pathlib import Path

import librosa
import torch
from e2k import P2K
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

try:
    from ipapy.arpabetmapper import ARPABETMapper
except AttributeError:
    sys.modules["collections"].MutableMapping = collections.abc.MutableMapping
    sys.modules["collections"].MutableSequence = collections.abc.MutableSequence
    from ipapy.arpabetmapper import ARPABETMapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
p2k = P2K()


def ipa_to_kana(ipa_str: str, ignore_unmapped: bool = True) -> list[str]:
    """IPAをカタカナ表記へ変換する"""
    logger.debug("IPA文字列をカナへ変換: %s", ipa_str)
    mapper = ARPABETMapper()
    result = mapper.map_unicode_string(
        ipa_str, ignore=ignore_unmapped, return_as_list=True
    )
    logger.debug("ARPAbet変換結果: %s", result)
    katakana = p2k(result)
    return katakana


def recognize_phonemes(audio_path: Path) -> str:
    """Wav2Vec2モデルで音素列を取得"""
    logger.info("音声を読み込み中: %s", audio_path)
    waveform, _sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    logger.info("モデル(%s)を読み込み", MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    input_values = feature_extractor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes_list = tokenizer.batch_decode(predicted_ids)
    phonemes = phonemes_list[0] if phonemes_list else ""
    logger.info("Recognized phonemes: %s", phonemes)
    return phonemes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="wav2vec2による音素列取得とカナ変換を行うユーティリティ"
    )
    parser.add_argument("audio_file", help="入力音声ファイルのパス")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error("音声ファイルが存在しません: %s", audio_path)
        sys.exit(1)

    phonemes = recognize_phonemes(audio_path)
    kana = ipa_to_kana(phonemes)
    logger.info("KANA: %s", kana)


if __name__ == "__main__":
    main()
