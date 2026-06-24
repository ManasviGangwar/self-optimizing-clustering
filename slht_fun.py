import numpy as np

def slht(s, idx, n, m, nk):
    """
    Calculates the Silhouette values for each cluster and the Global Silhouette index.
    """
    S_const = np.zeros(nk)
    
    # Sum up individual silhouette values for data points in each cluster
    for r in range(n):
        S_const[idx[r]] += s[r]
        
    S = np.zeros(nk)
    for j in range(nk):
        if m[j] != 0:
            S[j] = (1.0 / m[j]) * S_const[j]
            
    # Calculate Global Silhouette value
    GS = (1.0 / nk) * np.sum(S)
    
    return S, GS
