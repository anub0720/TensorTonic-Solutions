import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    # Write code here
    k=np.array(k)
    prob=(1-p)**(k-1)*p
    return {"pmf":prob,"mean":1/p}