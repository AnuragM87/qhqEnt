import numpy as np


def depolarization_compensation(rho, p=None):
    """
    Invert depolarizing noise: ρ_noisy = (1-p)·ρ_ideal + p·I/4
    Returns cleaned density matrix with higher purity.

    Parameters
    ----------
    rho : 4x4 density matrix (noisy)
    p   : noise fraction. If None, estimated from eigenvalues.

    Returns
    -------
    rho_clean : denoised density matrix
    p_est     : estimated (or given) noise fraction
    """
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    d = rho.shape[0]  # dimension (4 for two qubits)

    if p is None:
        # Estimate p from the smallest eigenvalue
        # For depolarizing noise: λ_min ≈ p / d
        eigvals = np.linalg.eigvalsh(rho)
        lambda_min = max(eigvals.min(), 0)
        p = lambda_min * d
        p = min(p, 0.99)  # cap to avoid division issues

    # Invert: ρ_clean = (ρ_noisy - p·I/d) / (1 - p)
    rho_clean = (rho - p * np.eye(d) / d) / (1 - p)

    # Fix numerical issues: enforce Hermitian, positive, trace-1
    rho_clean = (rho_clean + rho_clean.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho_clean)
    vals[vals < 0] = 0
    rho_clean = vecs @ np.diag(vals) @ vecs.conj().T
    rho_clean = rho_clean / np.trace(rho_clean)

    print(f"[Depolarization] Estimated noise p = {p:.4f}")
    print(f"[Depolarization] Purity before: {np.real(np.trace(rho @ rho)):.4f}")
    print(f"[Depolarization] Purity after:  {np.real(np.trace(rho_clean @ rho_clean)):.4f}")

    return rho_clean, p


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
    # eigh returns ascending order, take the largest `rank` eigenvalues
    idx = np.argsort(vals)[::-1][:rank]
    vals_keep = vals[idx]
    vecs_keep = vecs[:, idx]

    # Reconstruct
    rho_clean = np.zeros_like(rho)
    for i in range(rank):
        v = vecs_keep[:, i]
        rho_clean += vals_keep[i] * np.outer(v, v.conj())

    rho_clean = rho_clean / np.trace(rho_clean)

    print(f"[EigenFilter] Kept top {rank} eigenstate(s)")
    print(f"[EigenFilter] Discarded eigenvalues: {np.sort(vals)[::-1][rank:]}")
    print(f"[EigenFilter] Purity after: {np.real(np.trace(rho_clean @ rho_clean)):.4f}")

    return rho_clean
