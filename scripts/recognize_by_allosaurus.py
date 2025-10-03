# -*- coding: utf-8 -*-
# 1) 音声→音素（Allosaurus）
from allosaurus.app import read_recognizer
import re
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="音声ファイルを音素経由でカタカナに変換")
    parser.add_argument("audio_file", help="音声ファイルのパス")
    parser.add_argument("--lang", default="eng", help="言語コード (デフォルト: eng)")
    
    args = parser.parse_args()
    
    rec = read_recognizer()  # 既定モデル（多言語対応）
    phones = rec.recognize(args.audio_file, lang_id=args.lang)  # 例: "ð ɪ s ɪ z ə t ɜː s t"
    # ↑ 単語に依存しない phone 列。連結/弱化/脱落も音として出る

    kana = phones_to_kana(phones)
    print("PHONES:", phones)
    print("KANA:", kana)


# 2) 音素→カタカナ（“聞こえ”重視の簡易変換）
#   連結や弱化は Allosaurus の出力に反映されるので、ここは主に写し方の工夫だけ
#   例: flap ɾ を「ラ/リ/ル/レ/ロ」に寄せず「ラ行っぽいタ行」→「ラ」「ル」など好みで
MAP_SEQ = [
    (r"tʃ", "チ"), (r"dʒ", "ジ"), (r"ʃ", "シ"), (r"ʒ", "ジ"),
    (r"ts", "ツ"), (r"tɕ", "チ"), (r"dz", "ヅ"),
]
MAP_ONE = {
    "p":"プ","b":"ブ","t":"ト","d":"ド","k":"ク","g":"グ","m":"ム","n":"ン","ŋ":"ング",
    "f":"フ","v":"ヴ","s":"ス","z":"ズ","h":"ハ","l":"ル","r":"ル","ɹ":"ル","ɾ":"ル",
    "w":"ウ","j":"イ", "θ":"ス","ð":"ズ",
    # 母音系（長さは後段で処理）
    "i":"イ","iː":"イー","ɪ":"イ","e":"エ","eɪ":"エイ","ɛ":"エ",
    "æ":"ア","aɪ":"アイ","aʊ":"アウ","ɑ":"ア","ʌ":"ア","ə":"ア",
    "ɜː":"アー","ɚ":"アー","ɝ":"アー",
    "oʊ":"オウ","ɔ":"オ","ɔɪ":"オイ",
    "u":"ウ","uː":"ウー","ʊ":"ウ",
    "ː":"ー","ˈ":"", "ˌ":""
}

def phones_to_kana(phones_str: str) -> str:
    s = phones_str.strip().lower()

    # 連続音（2文字以上の合字）を先に
    for pat, rep in MAP_SEQ:
        s = re.sub(pat, rep, s)

    # トークン単位で処理（Allosaurus はスペース区切り）
    out = []
    tokens = s.split()
    for i,ph in enumerate(tokens):
        # 長音記号付きのセットを先に
        if ph in MAP_ONE:
            out.append(MAP_ONE[ph])
            continue

        # 長音化: 同じ母音が続く/ː が来たら「ー」を挿入
        if ph.endswith("ː"):
            base = ph.replace("ː","")
            out.append(MAP_ONE.get(base, "")) ; out.append("ー"); continue

        # ざっくり置換
        rep = MAP_ONE.get(ph, "")
        if rep:
            out.append(rep)
        # 無変換トークンは捨てる（句読点類など）
    kana = "".join(out)

    # 調音に基づく微整形（日本語っぽく）
    # 例1: 無声破裂直前の促音化（…ッ＋[p t k]）
    kana = re.sub(r"(ア|イ|ウ|エ|オ|ャ|ュ|ョ)?(プ|ト|ク)", r"ッ\2", kana)
    # 例2: 語中の n + [pbm] → 「ン（唇音）」はそのままでも可
    # 例3: 語境界を中点で見せたい場合
    kana = re.sub(r"\s+", "・", kana)

    # 仕上げ：全角カタカナ以外を除去
    kana = re.sub(r"[^\u30A0-\u30FF・ー]", "", kana)
    return kana

if __name__ == "__main__":
    main()