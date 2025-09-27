import re

import jamorasep
import pyopenjtalk
import tqdm

from datasets import load_dataset


def text_to_katakana(text: str) -> list[str]:
    """
    Convert Japanese text to katakana using pyopenjtalk.

    Args:
      text (str): Input Japanese text

    Returns:
      list[str]: List of katakana blocks
    """
    kana = pyopenjtalk.g2p(text, kana=True)
    # kanaは文字列として返されるので、型を明示
    if isinstance(kana, str):
        # カナの塊（カタカナのみ）をすべて抽出
        kana_blocks = re.findall(r"[ァ-ンヴー]+", kana)
        return kana_blocks
    else:
        # リストが返された場合（通常はあり得ないがsafety check）
        return []


def get_livedoor_texts() -> list[str]:
    dataset = load_dataset("shunk031/livedoor-news-corpus", split=None)
    all_texts = []
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            all_texts.append(item["content"])  # type: ignore
    return all_texts


def split_text(text: str, length: int) -> list[str]:
    """
    Split text into chunks of a specified length.

    Args:
      text (str): Input text
      length (int): Length of each chunk

    Returns:
      list[str]: List of text chunks
    """
    # AIDEV-NOTE: textをスペースか記号で分割
    tokens = re.split(r"[\s　、。！？\n\r\t,.!?;:・「」（）【】『』…―—]+", text)

    tokens = [t + "、" for t in tokens if t]
    tokens[-1] = tokens[-1][:-1]
    chunks = []
    current = ""
    for token in tokens:
        if not token:
            continue
        # tokenがlengthを超える場合はそのままchunkに追加
        if len(token) > length:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(token)
            continue
        # 次のtokenを追加してもlengthを超えない場合はcurrentに追加
        if len(current) + len(token) <= length:
            current += token
        else:
            if current:
                chunks.append(current)
            current = token
    if current:
        chunks.append(current)
    return chunks


def text_to_kana(text: str) -> list[str]:
    """
    Convert Japanese text to katakana using pyopenjtalk.

    Args:
      text (str): Input Japanese text

    Returns:
      str: Converted katakana string
    """

    kana = pyopenjtalk.g2p(text, kana=True)
    # kanaは文字列として返されるので、型を明示
    kana_blocks = re.findall(r"[ァ-ンヴー]+", kana) if isinstance(kana, str) else []

    return kana_blocks


def split_to_moras(kana: str) -> list[str]:
    """
    Split a katakana string into moras using jamorasep.

    Args:
      kana (str): Input katakana string

    Returns:
      list[str]: List of moras
    """
    kanamap = {"クヮ": "クァ", "ヂ": "ジ", "ヅ": "ズ", "ヱ": "エ", "ヲ": "オ"}
    for src, target in kanamap.items():
        kana = kana.replace(src, target)
    return jamorasep.parse(kana, output_format="katakana")


def corpus_to_mora(texts: list[str]) -> list[str]:
    """
    Convert a list of texts to a list of mora lists.

    Args:
      texts (list[str]): List of input texts

    Returns:
      list[list[str]]: List of mora lists
    """
    all_moras = ["<SEP>"]
    for text in tqdm.tqdm(texts):
        sentences = split_text(text, 2500)
        for sentence in sentences:
            kana_blocks = text_to_kana(sentence)
            for block in kana_blocks:
                moras = split_to_moras(block)
                if moras:
                    all_moras += moras
                    all_moras.append("<SEP>")
    return all_moras
