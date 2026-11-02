def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    stopwords_set = set(stopwords)
    return [token for token in tokens if token not in stopwords_set]