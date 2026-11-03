import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    y_train=np.array(y_train)
    labels, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)

    candidates = labels[counts == max_count]
    
    for val in y_train:
        if val in candidates:
            majority = val
            break
    
    predictions = np.full(len(X_test), majority, dtype=int)
    return predictions