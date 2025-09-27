from collections import Counter, defaultdict


def mora_unigram_frequency(moras: list[str]) -> dict[str, int]:
    """
    Calculate the frequency of each mora in the list.

    Args:
      moras (list[str]): List of moras

    Returns:
      dict[str, int]: Dictionary with mora as key and its frequency as value
    """
    frequency = Counter(moras)
    frequency.pop("<SEP>", None)  # Remove <SEP> from frequency count
    return dict(frequency)


def mora_bigram_frequency(moras: list[str]) -> dict[str, dict[str, int]]:
    """
    Calculate the frequency of each mora bigram in a list of Japanese texts.
    Uses a special token <SEP> for text boundaries and handles chunk boundaries.

    Args:
        texts (list[str]): List of Japanese texts

    Returns:
        dict[tuple[str, str], float]: Dictionary of mora bigrams
            and their appearance rate
    """
    # Generate bigrams
    bigrams = []
    for i in range(len(moras) - 1):
        bigram = (moras[i], moras[i + 1])
        bigrams.append(bigram)

    bigram_freq = Counter(bigrams)

    # Convert to probabilities
    result_dict = {}
    for (m1, m2), count in bigram_freq.items():
        if m1 not in result_dict:
            result_dict[m1] = {}
        result_dict[m1][m2] = count
    return result_dict


# Additional utility function to get conditional probabilities
def get_conditional_probabilities(
    bigram_freq: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """
    Convert bigram frequencies to conditional probabilities P(w2|w1).

    Args:
        bigram_freq (dict[str, dict[str, int]]): Dictionary of bigram frequencies

    Returns:
        dict[str, dict[str, float]]: Dictionary where first key is w1, second key is w2,
                                     value is P(w2|w1)
    """
    # Calculate marginal frequencies for first words
    first_word_freq = defaultdict(int)
    for w1, w1_dict in bigram_freq.items():
        for _w2, freq in w1_dict.items():
            first_word_freq[w1] += freq

    # Calculate conditional probabilities
    conditional_probs = defaultdict(dict)
    for w1, w1_dict in bigram_freq.items():
        for w2, freq in w1_dict.items():
            conditional_probs[w1][w2] = (
                freq / first_word_freq[w1] if first_word_freq[w1] > 0 else 0.0
            )

    return dict(conditional_probs)
