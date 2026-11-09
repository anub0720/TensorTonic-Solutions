def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    X=np.array(X)
    W=np.array(W)
    out=X@W+b
    return out.tolist()