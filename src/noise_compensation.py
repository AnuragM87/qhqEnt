import numpy as np

def eigenvalue_filter(rho, rank=1):
    """
    Keep only the top `rank` eigenstates, discard noise.

    Parameters
    ----------
    rho  : 4x4 density matrix
    rank : number of eigenstates to keep (1 = pure state)

    Returns
    -------
    rho_clean : filtered density matrix
    """
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    vals, vecs = np.linalg.eigh(rho)
    idx = np.argsort(vals)[::-1][:rank]
    vals_keep = vals[idx]
    vecs_keep = vecs[:, idx]

    rho_clean = np.zeros_like(rho)
    for i in range(rank):
        v = vecs_keep[:, i]
        rho_clean += vals_keep[i] * np.outer(v, v.conj())

    rho_clean = rho_clean / np.trace(rho_clean)

    return rho_clean
