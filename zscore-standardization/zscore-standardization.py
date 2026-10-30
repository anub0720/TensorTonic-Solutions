import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X,
    forcing output to zero for slices where standard deviation <= eps.
    """
    arr = np.array(X)
    
    # Compute mean and standard deviation along the specified axis
    mu = np.mean(arr, axis=axis, keepdims=True)
    sigma = np.std(arr, axis=axis, keepdims=True)
    
    # 1. Decide per slice whether sigma > eps (boolean mask)
    valid_sigma = sigma > eps
    
    # 2 & 3. Use sigma where valid, or a safe dummy value (like 1.0) to avoid division warnings
    safe_sigma = np.where(valid_sigma, sigma, 1.0)
    
    # Perform the division
    z = (arr - mu) / safe_sigma
    
    # 4. Force the output to zero for the slices where sigma <= eps
    z = np.where(valid_sigma, z, 0.0)
    
    return z