from kanagib.voicevox import VoiceVoxClient
from kanagib import generate_by_bigram
from kanagib import generate_by_unigram
import json
import os
from pathlib import Path
import uuid
import tqdm
VOICE_VOX_SPEAKER = 3
VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://localhost:50021")
INPUT = "gibberish.txt"
OUTPUT_DIR = "output"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate gibberish text and synthesize speech using VOICEVOX")
    parser.add_argument("--input_file", type=str, default=INPUT, help="Input text file path")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for WAV files and JSON")
    parser.add_argument("--voicevox_url", type=str, default=VOICEVOX_URL, help="VOICEVOX server URL")
    parser.add_argument("--voicevox_speaker", type=int, default=VOICE_VOX_SPEAKER, help="VOICEVOX speaker ID")
    args = parser.parse_args()
    voicevox_client = VoiceVoxClient(args.voicevox_url)

    output_wav_dir = Path(args.output_dir) / "wav"
    output_json_path = Path(args.output_dir) / "gibberish_with_wav.json"

    output_wav_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    

    voicevox_client = VoiceVoxClient(args.voicevox_url)
    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(texts)} lines from {args.input_file}")

    result_json = []
    for text in tqdm.tqdm(texts):

        id = str(uuid.uuid4())
        wav_filename = f"gibberish_{id}_{text[:100]}.wav"
        content, duration = voicevox_client.generate_wav(text, 
                                                         speaker=args.voicevox_speaker, 
                                                         filepath=os.path.join(output_wav_dir, 
                                                                               wav_filename))
        result_json.append({
            "id": id,
            "text": text,
            "wav_file": wav_filename,
            "wav_speaker": args.voicevox_speaker,
            "wav_duration": duration
        })
        print(f"Saved to {wav_filename}, duration: {duration:.2f} seconds")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"All data saved to {output_json_path}")
    
      