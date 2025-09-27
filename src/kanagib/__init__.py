"""Japanese gibberish generation using mora-based statistical models."""

from .gibberish_generator import (
    UnigramGenerator,
    BigramGenerator,
)

generate_by_unigram = UnigramGenerator().generate
generate_by_bigram = BigramGenerator().generate

__all__ = [
    "generate_by_unigram",
    "generate_by_bigram",
]