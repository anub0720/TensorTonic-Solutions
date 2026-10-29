import numpy as np
def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    recommended=np.array(recommended)
    relevant=np.array(relevant)
    topk=recommended[:k]
    ans=np.intersect1d(topk,relevant)
    precision=ans.size/k
    recall=ans.size/relevant.size
    return [precision,recall]