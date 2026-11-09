import numpy as np

def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    points = np.array(points)     # Shape: (N, D) -> (4, 2)
    centroids = np.array(centroids) # Shape: (K, D) -> (2, 2)
    
    # 1. Expand dimensions for broadcasting:
    # points[:, np.newaxis, :] has shape (4, 1, 2)
    # centroids[np.newaxis, :, :] has shape (1, 2, 2)
    # Their difference has shape (4, 2, 2)
    diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    
    # 2. Square, sum over the feature dimension (axis=2), resulting in shape (4, 2)
    distances = np.sum(diff ** 2, axis=2)
    
    # 3. Find the index of the minimum centroid for each point along axis=1
    j = np.argmin(distances, axis=1)
    
    return j.tolist()