# pip install -U torch torchaudio transformers librosa unidecode
# モデルは16kHz想定。手元のWAVを16kHz, monoにしておくと安定します。

import torch, librosa, re, sys
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# CLIから音声ファイルパスを取得
if len(sys.argv) < 2:
    print("Usage: python recognize_by_wav2vec2phonome.py <audio_file_path>")
    sys.exit(1)

path = sys.argv[1]

# 1) 音素認識モデルをロード（より一般的なモデルを使用）
MODEL_ID = "facebook/wav2vec2-base-960h"  # 安定した英語音声認識モデル
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

# 2) 音声読み込み（16kHz, mono）
wav, sr = librosa.load(path, sr=16000, mono=True)

# 長尺は分割。ここでは10秒チャンク＋ストライドで簡易的にマージ
def chunk_iter(x, rate=16000, chunk_s=10.0, stride_s=1.0):
    cs, ss = int(chunk_s*rate), int(stride_s*rate)
    i=0
    while i < len(x):
        yield x[i:i+cs]
        if i+cs >= len(x): break
        i += cs - ss

# 3) 逐次デコード（CTCで文字列を得る）
def decode_logits(logits):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

transcriptions=[]
with torch.no_grad():
    for seg in chunk_iter(wav):
        inputs = processor(seg, sampling_rate=16000, return_tensors="pt")
        logits = model(**inputs).logits
        transcriptions.append(decode_logits(logits))

transcription = " ".join(transcriptions)
print("Recognized:", transcription)

# 4) 英語 → カタカナ（簡易ルール。必要に応じて強化）

# 4) IPA → カタカナ（“聞こえ”優先の簡易ルール。必要に応じて強化）
#   連結・弱化はIPA側に出たものを尊重し、日本語らしい整形だけ行う。
SEQ = [
    (r"tʃ", "チ"), (r"dʒ", "ジ"), (r"ʃ", "シ"), (r"ʒ", "ジ"),
    (r"ts", "ツ"), (r"tɕ", "チ"), (r"dz", "ヅ"),
    (r"ɾ", "ラ"),  # flapをラ行寄りに（好みで「ル」「ッ」等に調整）
]
ONE = {
    "p":"プ","b":"ブ","t":"ト","d":"ド","k":"ク","g":"グ",
    "m":"ム","n":"ン","ŋ":"ング","f":"フ","v":"ヴ",
    "s":"ス","z":"ズ","h":"ハ","l":"ル","r":"ル","ɹ":"ル",
    "w":"ウ","j":"イ","θ":"ス","ð":"ズ",

    # 母音系（長短は「ː」で後処理）
    "i":"イ","ɪ":"イ","e":"エ","ɛ":"エ","æ":"ア","a":"ア",
    "ɑ":"ア","ʌ":"ア","ə":"ァ","ɜ":"ア","ɚ":"ァ","ɝ":"アー",
    "o":"オ","ɔ":"オ","u":"ウ","ʊ":"ウ",
    "eɪ":"エイ","aɪ":"アイ","ɔɪ":"オイ","oʊ":"オウ","aʊ":"アウ",
    "ˈ":"", "ˌ":"", "ː":"ー"
}

def ipa_to_kana(ipa_text: str) -> str:
    s = ipa_text.lower()
    # 合字から先に
    for pat,rep in SEQ:
        s = re.sub(pat, rep, s)
    # トークン単位（空白区切り想定）
    out=[]
    for ph in s.split():
        if ph in ONE:
            out.append(ONE[ph])
        elif ph.endswith("ː") and ph[:-1] in ONE:
            out.append(ONE[ph[:-1]] + "ー")
        else:
            # 未知は軽くスキップ（必要ならログ/可視化）
            pass
    kana = "".join(out)

    # 日本語っぽい整形（例：無声破裂直前の促音化）
    kana = re.sub(r"(ッ)?(プ|ト|ク)([ァィゥェォャュョ]?)", r"ッ\2\3", kana)
    # 連続母音の長音化簡略
    kana = re.sub(r"(ア|イ|ウ|エ|オ)\1+", r"\1ー", kana)
    return kana

# 4) 英語 → カタカナ（簡易ルール。必要に応じて強化）
def text_to_kana(text: str) -> str:
    # 英語の音を日本語カタカナに変換する簡易ルール
    s = text.lower()
    
    # 基本的な音素マッピング
    replacements = [
        (r'\bthe\b', 'ザ'), (r'\band\b', 'アンド'), (r'\bof\b', 'オブ'),
        (r'ch', 'チ'), (r'sh', 'シ'), (r'th', 'ス'), 
        (r'ng', 'ング'), (r'ck', 'ク'),
        ('a', 'ア'), ('e', 'エ'), ('i', 'イ'), ('o', 'オ'), ('u', 'ウ'),
        ('b', 'ブ'), ('c', 'ク'), ('d', 'ド'), ('f', 'フ'), ('g', 'グ'),
        ('h', 'ハ'), ('j', 'ジ'), ('k', 'ク'), ('l', 'ル'), ('m', 'ム'),
        ('n', 'ン'), ('p', 'プ'), ('q', 'ク'), ('r', 'ル'), ('s', 'ス'),
        ('t', 'ト'), ('v', 'ヴ'), ('w', 'ワ'), ('x', 'クス'), ('y', 'イ'), ('z', 'ズ')
    ]
    
    for eng, kata in replacements:
        s = re.sub(eng, kata, s)
    
    # 英語以外の文字を除去
    s = re.sub(r'[^ァ-ヶー]', '', s)
    
    return s

kana = text_to_kana(transcription)
print("KANA:", kana)