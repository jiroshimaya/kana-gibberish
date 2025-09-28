from pathlib import Path

import tqdm

from kanagib import generate_by_bigram, generate_by_random, generate_by_unigram

TEXT_LENGTH = 32
MIN_LENGTH = 1
NUM_SENTENCES = 5
OUTPUT = "output/gibberish.txt"
MODE = "bigram"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate gibberish text and synthesize speech using VOICEVOX"
    )
    parser.add_argument(
        "--mode",
        choices=["unigram", "bigram", "random"],
        default="bigram",
        help="Gibberish generation mode",
    )
    parser.add_argument(
        "--stop_on_sep",
        action="store_true",
        help=(
            "Stop generation when a separator mora is encountered "
            "(only for bigram mode)"
        ),
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=NUM_SENTENCES,
        help="Number of sentences to generate",
    )
    parser.add_argument(
        "--text_length",
        type=int,
        default=TEXT_LENGTH,
        help="Length of each generated text",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=MIN_LENGTH,
        help="Minimum length of each generated text",
    )

    parser.add_argument(
        "--output", type=str, default=OUTPUT, help="Output text file path"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_text() -> str:
        if args.mode == "bigram":
            if args.stop_on_sep:
                return generate_by_bigram(
                    args.text_length,
                    avoid_sep=False,
                    stop_on_sep=True,
                    min_length=args.min_length,
                )
            else:
                return generate_by_bigram(args.text_length, avoid_sep=True)
        elif args.mode == "random":
            return generate_by_random(args.text_length)
        else:
            return generate_by_unigram(args.text_length)

    with output_path.open("w", encoding="utf-8") as f:
        for _ in tqdm.tqdm(range(args.num_sentences)):
            text = generate_text()
            f.write(text + "\n")
    print("Done.")
