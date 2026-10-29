import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    s=np.array(s,dtype=float)
    w=np.array(w,dtype=float)
    g=np.array(g,dtype=float)
    st=beta*s+(1-beta)*(g**2)
    wt=w-(lr/(np.sqrt(st+eps)))*g
    return wt,st