import numpy as np
def interaction_features(X: list) -> list:
    """
    Returns original features followed by unique pairwise products.
    """
    # Write code here
    X=np.array(X)
    d=X.shape[1]
    i, j = np.triu_indices(d, k=1)
    interactions = X[:, i] * X[:, j]
    result = np.concatenate((X, interactions), axis=1)
    return result.tolist()