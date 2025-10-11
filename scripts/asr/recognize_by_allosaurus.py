# 1) 音声→音素（Allosaurus）
import argparse
import collections
import collections.abc
import logging
import sys

from allosaurus.app import read_recognizer
from e2k import P2K

try:
    from ipapy.arpabetmapper import ARPABETMapper
except AttributeError:
    sys.modules["collections"].MutableMapping = collections.abc.MutableMapping
    sys.modules["collections"].MutableSequence = collections.abc.MutableSequence
    from ipapy.arpabetmapper import ARPABETMapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

p2k = P2K()


def ipa_to_kana(ipa_str: str, ignore_unmapped: bool = True) -> list[str]:
    """
    IPA → ARPAbet（リスト形式）変換（ipapy を使う方法）
    未対応の文字を無視するオプションあり。
    例: "pɹuːf" → ["P", "R", "UW", "F"]
    """
    logger.debug("IPA文字列をカナへ変換: %s", ipa_str)
    mapper = ARPABETMapper()
    # return_as_list=True にすると、分割されたリストを得られる
    result = mapper.map_unicode_string(
        ipa_str, ignore=ignore_unmapped, return_as_list=True
    )
    logger.debug("ARPAbet変換結果: %s", result)
    katakana = p2k(result)
    return katakana


def main():
    logger.info("Allosaurusを用いた音素変換を実行します")
    parser = argparse.ArgumentParser(
        description="音声ファイルを音素経由でカタカナに変換"
    )
    parser.add_argument("audio_file", help="音声ファイルのパス")
    parser.add_argument("--lang", default="eng", help="言語コード (デフォルト: eng)")

    args = parser.parse_args()

    rec = read_recognizer()  # 既定モデル（多言語対応）
    phones = rec.recognize(
        args.audio_file, lang_id=args.lang
    )  # 例: "ð ɪ s ɪ z ə t ɜː s t"
    # ↑ 単語に依存しない phone 列。連結/弱化/脱落も音として出る

    kana = ipa_to_kana(phones)
    logger.info("PHONES: %s", phones)
    logger.info("KANA: %s", kana)


if __name__ == "__main__":
    main()
