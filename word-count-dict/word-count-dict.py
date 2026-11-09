from collections import Counter
def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    # Write code here
    flat_words = [word for sentence in sentences for word in sentence]
    word_counts = Counter(flat_words)
    return dict(word_counts)