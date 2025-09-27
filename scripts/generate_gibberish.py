from kanagib.voicevox import VoiceVoxClient
from kanagib import generate_by_bigram
from kanagib import generate_by_unigram
import os
import tqdm
TEXT_LENGTH = 32
NUM_SENTENCES = 5
OUTPUT = "output/gibberish.txt"
MODE = "bigram"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate gibberish text and synthesize speech using VOICEVOX")
    parser.add_argument("--mode", choices=["unigram", "bigram"], default="bigram", help="Gibberish generation mode")
    parser.add_argument("--num_sentences", type=int, default=NUM_SENTENCES, help="Number of sentences to generate")
    parser.add_argument("--text_length", type=int, default=TEXT_LENGTH, help="Length of each generated text")
    parser.add_argument("--output", type=str, default=OUTPUT, help="Output text file path")
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.mode == "bigram":
        generator = lambda: generate_by_bigram(args.text_length, avoid_sep=True)
    else:
        generator = lambda: generate_by_unigram(args.text_length)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in tqdm.tqdm(range(args.num_sentences)):
            text = generator()
            f.write(text + "\n")
    print("Done.")
      