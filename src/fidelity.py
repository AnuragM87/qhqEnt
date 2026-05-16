
import numpy as np

def uhlmann_fidelity(rho, sigma, eps=1e-12):
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    rho = rho / np.trace(rho)
    sigma = sigma / np.trace(sigma)

    def clean(mat):
        vals, vecs = np.linalg.eigh(mat)
        vals[vals < eps] = 0
        return vecs @ np.diag(vals) @ vecs.conj().T

    rho = clean(rho)
    sigma = clean(sigma)

    eigvals = np.linalg.eigvals(rho @ sigma)

    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0

    F = (np.sum(np.sqrt(eigvals)))**2
    return float(np.clip(F, 0.0, 1.0))

