import numpy as np

def lagrangepoly(X, Y):
    """
    Lagrange interpolation polynomial fitting a set of points.
    Returns the coefficients P, the roots of the derivative R, 
    and the evaluated Y values at those roots S.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N = len(X)
    
    pvals = np.zeros((N, N))
    
    for i in range(N):
        # Create array of all X values except X[i]
        X_ex = np.concatenate((X[:i], X[i+1:]))
        pp = np.poly(X_ex)
        # Scale so value is exactly 1 at X[i]
        pvals[i, :] = pp / np.polyval(pp, X[i])
        
    P = np.dot(Y, pvals)
    
    # Calculate roots of the derivative (extrema)
    dP = np.polyder(P)
    R = np.roots(dP)
    
    # Calculate actual values at points of zero derivative
    S = np.polyval(P, R)
    
    return P, R, S
