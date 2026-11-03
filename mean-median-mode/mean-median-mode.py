from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x=np.array(x,dtype=float)
    mean=float(np.mean(x))
    median=float(np.median(x))
    counter=Counter(x)
    mode_value, mode_count = counter.most_common(1)[0]
    return {"mean": mean, "median": median, "mode": float(mode_value)}