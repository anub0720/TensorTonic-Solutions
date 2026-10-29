import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    x=np.array(x,dtype=float)
    p=np.array(p,dtype=float)
    ex=np.sum(x*p)
    return ex