import numpy as np
from soc import soc
from slht import slht
from lagrangepoly import lagrangepoly
from sklearn.metrics import silhouette_samples

def factorcal(x, nk, iter_max=10):
    """
    Computes the optimization factor via Lagrange interpolation.
    """
    flag = 0
    factor = np.ones((10, nk))
    GS = np.zeros(10)
    
    for iter_idx in range(iter_max):
        print(f"Optimizing Iteration: {iter_idx+1}/{iter_max}")
        result = soc(x, nk, factor[iter_idx, :])
        
        # Break if any cluster is empty
        if np.min(result['m']) == 0:
            flag = 1
            break
            
        # Break if threshold distances duplicate
        for g in range(nk - 1):
            for gg in range(g + 1, nk):
                if result['d1'][g] == result['d1'][gg]:
                    flag = 1
                    
        if flag == 1:
            break
            
        # Calculate silhouette values using sklearn
        # silhouette_samples expects 2D data and labels
        s = silhouette_samples(x, result['idx'])
        S, GS[iter_idx] = slht(s, result['idx'], result['n'], result['m'], nk)
        
        # Fit Lagrange polynomial
        P, r, _ = lagrangepoly(result['d1'], S)
        
        polym = P.copy()
        # Equivalent to polym(1,nk)=polym(1,nk)-1 in MATLAB
        polym[-1] = polym[-1] - 1  
        r_roots = np.roots(polym)
        
        sumn = np.zeros_like(r_roots, dtype=complex)
        for i in range(len(r_roots)):
            for j in range(nk):
                sumn[i] += polym[nk - 1 - j] * (r_roots[i]**j)
                
        label = np.argmin(np.abs(sumn))
        dmax = np.abs(r_roots[label])
        
        if iter_idx + 1 < 10:
            factor[iter_idx + 1, :] = dmax / result['d1']
            
    # Find max Global Silhouette to assign final factor
    if flag == 1 and iter_idx > 0:
        label_max = np.argmax(GS[:iter_idx])
    else:
        label_max = np.argmax(GS[:iter_idx + 1])
        
    fac = factor[label_max, :]
    return fac
