import numpy as np

def soc(x, nk, factor):
    """
    Implements the Improved Mountain Clustering Technique (IMC)
    used for SOC. Accepts an array of factors for threshold calculation.
    """
    n, k_dim = x.shape
    
    # Normalize each dimension of hyperspace
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    z = x_max - x_min
    
    u = np.zeros((n, k_dim))
    for j in range(n):
        y = x[j, :] - x_min
        # Handle division by zero
        u[j, :] = np.divide(y, z, out=np.zeros_like(y), where=z!=0)
            
    U = u.copy()
    
    m = np.zeros(nk, dtype=int)
    t = np.zeros(nk + 1, dtype=int)
    t[0] = n
    
    sl = np.zeros((n, nk + 1), dtype=int)
    sl[:, 0] = np.arange(n)
    
    d1 = np.zeros(nk)
    P = np.zeros((n, nk))
    cc_norm = np.zeros((nk, k_dim))
    
    SL = np.zeros((n, nk), dtype=int) - 1  # -1 indicates empty
    clst = np.zeros((n, k_dim, nk))
    c_disp = np.zeros((n, k_dim, nk + 1)) + 255
    
    for v in range(nk):
        if t[v] != 0:
            d_val = 0.0
            for j in range(t[v]):
                row_sum = np.sum(x[sl[j, v], :])
                if row_sum != 0:
                    d_val += np.min(x[sl[j, v], :]) / row_sum
            
            # Use array factor indexing
            d1[v] = ((1.0 / (2.0 * t[v])) * d_val) * factor[v]
            
            # Calculate potential value using mountain function
            for r in range(t[v]):
                idx_r = sl[r, v]
                diffs = u[idx_r, :] - u[sl[:t[v], v], :]
                dist_sq = np.sum(diffs**2, axis=1)
                P[r, v] = np.sum(np.exp(-dist_sq / (d1[v]**2)))
                
            # Select cluster center
            zmax = np.argmax(P[:t[v], v])
            cc_norm[v, :] = u[sl[zmax, v], :]
            
            # Assign points to cluster
            t_next = 0
            for r in range(t[v]):
                idx_r = sl[r, v]
                dist = np.sum((u[idx_r, :] - cc_norm[v, :])**2)
                
                if dist <= d1[v]:
                    SL[m[v], v] = idx_r
                    clst[m[v], :, v] = x[idx_r, :]
                    c_disp[idx_r, :, v] = x[idx_r, :]
                    m[v] += 1
                else:
                    sl[t_next, v + 1] = idx_r
                    c_disp[idx_r, :, v] = 255
                    t_next += 1
            t[v + 1] = t_next

    # Distribute left out data points
    if t[nk] != 0:
        for r_idx in range(t[nk]):
            idx_r = sl[r_idx, nk]
            diffs = U[idx_r, :] - cc_norm
            D = np.sum(diffs**2, axis=1)
            v_min = np.argmin(D)
            
            SL[m[v_min], v_min] = idx_r
            clst[m[v_min], :, v_min] = x[idx_r, :]
            c_disp[idx_r, :, v_min] = x[idx_r, :]
            m[v_min] += 1

    # Calculate idx labels and distances
    idx = np.zeros(n, dtype=int)
    dd = np.zeros((n, nk))
    
    for r in range(n):
        for v in range(nk):
            if m[v] != 0:
                dd[r, v] = np.sum((U[r, :] - cc_norm[v, :])**2)
                if r in SL[:m[v], v]:
                    idx[r] = v
                    
    # Partition matrix
    part = np.zeros((n, nk))
    for j in range(n):
        label_min = np.argmin(dd[j, :])
        part[j, label_min] = 1
        
    return {
        'dd': dd, 'part': part, 'cc_norm': cc_norm, 
        'idx': idx, 'c_disp': c_disp, 'clst': clst, 
        'm': m, 'SL': SL, 'n': n, 'd1': d1
    }
