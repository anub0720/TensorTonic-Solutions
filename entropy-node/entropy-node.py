
import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns Shannon entropy from class labels as a Python float.
    """
    y = np.array(y)

    # Count occurrences of each class
    _, counts = np.unique(y, return_counts=True)

    # Convert counts to probabilities
    p = counts / len(y)

    # Calculate entropy (ignore zero probabilities)
    p = p[p > 0]
    hs = -np.sum(p * np.log2(p))

    return float(hs)
