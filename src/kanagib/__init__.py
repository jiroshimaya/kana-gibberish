"""Japanese gibberish generation using mora-based statistical models."""

from .gibberish_generator import (
    BigramGenerator,
    UnigramGenerator,
)

generate_by_unigram = UnigramGenerator().generate
generate_by_bigram = BigramGenerator().generate

__all__ = [
    "generate_by_bigram",
    "generate_by_unigram",
]
