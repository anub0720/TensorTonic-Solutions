def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    # Write code here
    result=[]
    for row in matrix:
        nz=[x for x in row if x!=0]
        mean=sum(nz)/len(nz) if(nz) else 0.0
        result.append([0.0 if x==0 else float(x-mean) for x in row])
    return result