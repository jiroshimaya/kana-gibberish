"""Japanese gibberish generation using mora-based statistical models."""

from .gibberish_generator import (
    generate_by_bigram,
    generate_by_random,
    generate_by_unigram,
)

__all__ = [
    "generate_by_bigram",
    "generate_by_random",
    "generate_by_unigram",
]
