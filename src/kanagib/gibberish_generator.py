import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnigramGenerator:
    """
    A generator for creating Japanese gibberish using unigram (frequency-based) statistics.
    """

    _DEFAULT_UNIGRAM_FILE = (
        Path(__file__).parent / "data" / "livedoor_mora_unigram_frequency.json"
    )

    def __init__(self, unigram_file: str | None = None):
        """
        Initialize the unigram generator.

        Args:
            unigram_file: Path to custom unigram frequency JSON file (optional)

        If no file is provided, default built-in data will be used.

        Raises:
            FileNotFoundError: If specified custom file is not found
        """
        logger.debug("Initializing UnigramGenerator")

        if unigram_file is not None:
            logger.debug(f"Loading custom unigram weights from {unigram_file}")
            file_path = Path(unigram_file)
        else:
            logger.debug("Loading default unigram weights")
            file_path = self._DEFAULT_UNIGRAM_FILE

        if not file_path.exists():
            raise FileNotFoundError(f"Unigram data file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            unigram_freq = json.load(f)

        # Pre-compute normalized probabilities for efficient generation
        self._moras = list(unigram_freq.keys())
        probs = np.array([unigram_freq[m] for m in self._moras], dtype=np.float64)
        self._probs = probs / probs.sum()  # normalize

        logger.info("UnigramGenerator initialized successfully")

    def generate(self, length: int) -> str:
        """
        Generate a random katakana sequence using mora frequency probabilities.

        Args:
          length (int): Length of the sequence

        Returns:
          str: Generated katakana sequence

        Raises:
          ValueError: If length is not positive
        """
        if length <= 0:
            raise ValueError(f"Expected positive integer, got {length}")

        sequence = "".join(np.random.choice(self._moras, size=length, p=self._probs))
        return sequence


class BigramGenerator:
    """
    A generator for creating Japanese gibberish using bigram (sequential) statistics.
    """

    _DEFAULT_BIGRAM_FILE = (
        Path(__file__).parent / "data" / "livedoor_mora_bigram_freqeuncy.json"
    )

    def __init__(self, bigram_file: str | None = None):
        """
        Initialize the bigram generator.

        Args:
            bigram_file: Path to custom bigram frequency JSON file (optional)

        If no file is provided, default built-in data will be used.

        Raises:
            FileNotFoundError: If specified custom file is not found
        """
        logger.debug("Initializing BigramGenerator")

        if bigram_file is not None:
            logger.debug(f"Loading custom bigram weights from {bigram_file}")
            file_path = Path(bigram_file)
        else:
            logger.debug("Loading default bigram weights")
            file_path = self._DEFAULT_BIGRAM_FILE

        if not file_path.exists():
            raise FileNotFoundError(f"Bigram data file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            bigram_freq = json.load(f)

        # Convert bigram frequencies to conditional probabilities
        self._conditional_probs = self._convert_bigram_to_conditional_probs(bigram_freq)

        logger.info("BigramGenerator initialized successfully")

    @staticmethod
    def _convert_bigram_to_conditional_probs(
        bigram_freq: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, float]]:
        """
        Convert bigram frequency counts to conditional probabilities.

        Args:
            bigram_freq: Bigram frequency dictionary {w1: {w2: count}}

        Returns:
            dict[str, dict[str, float]]: Conditional probabilities {w1: {w2: P(w2|w1)}}
        """
        logger.debug("Converting bigram frequencies to conditional probabilities")
        conditional_probs = {}

        for w1, w2_counts in bigram_freq.items():
            total_count = sum(w2_counts.values())
            if total_count > 0:
                conditional_probs[w1] = {
                    w2: count / total_count for w2, count in w2_counts.items()
                }
            else:
                conditional_probs[w1] = {}

        return conditional_probs

    def generate(
        self, max_length: int = 100, stop_on_sep: bool = True, avoid_sep: bool = False
    ) -> str:
        """
        Generate a random mora sequence using bigram conditional probabilities.
        Starts with <SEP> token and continues until <SEP> token or max_length is reached.

        Args:
            max_length: Maximum length of generated sequence (excluding special tokens)
            stop_on_sep: Whether to stop generation when <SEP> token is encountered
            avoid_sep: If True, excludes <SEP> from candidate tokens during generation,
                      ensuring generation continues until max_length

        Returns:
            str: Generated mora sequence (without special tokens)

        Raises:
            ValueError: If max_length is not positive or conditional_probs is empty
        """
        if max_length <= 0:
            raise ValueError(f"Expected positive integer, got {max_length}")

        if not self._conditional_probs:
            raise ValueError("Conditional probabilities dictionary is empty")

        # Start with SEP token
        current_mora = "<SEP>"
        sequence = []

        for _ in range(max_length):
            # Get possible next moras for current mora
            if current_mora not in self._conditional_probs:
                # If current mora not found, break (shouldn't happen with proper training data)
                print(
                    f"Warning: '{current_mora}' not found in conditional probabilities"
                )
                break

            next_mora_probs = self._conditional_probs[current_mora]

            # Filter out <SEP> token if avoid_sep is True
            if avoid_sep and "<SEP>" in next_mora_probs:
                filtered_probs = {
                    k: v for k, v in next_mora_probs.items() if k != "<SEP>"
                }
                if filtered_probs:  # Ensure we have at least one option
                    # Renormalize probabilities after removing <SEP>
                    total_prob = sum(filtered_probs.values())
                    next_mora_probs = {
                        k: v / total_prob for k, v in filtered_probs.items()
                    }
                # If all options were <SEP>, keep original probabilities (fallback)

            # Convert to lists for numpy choice
            next_moras = list(next_mora_probs.keys())
            probs = list(next_mora_probs.values())

            # Choose next mora based on probabilities
            next_mora = np.random.choice(next_moras, p=probs)

            # Check if we reached end of sequence
            if next_mora == "<SEP>":
                if stop_on_sep:
                    break
                # If stop_on_sep is False, skip <SEP> and continue with current mora
                continue

            # Add to sequence and update current mora
            sequence.append(next_mora)
            current_mora = next_mora

        return "".join(sequence)
