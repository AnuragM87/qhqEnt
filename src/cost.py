import numpy as np
from .qhqstates import U_total
from scipy.linalg import sqrtm

def uhlmann_fidelity(rho, sigma):
    """
    Compute Uhlmann fidelity between two density matrices.

    F(ρ,σ) = (Tr sqrt( sqrt(ρ) σ sqrt(ρ) ))^2
    """

    # Ensure Hermitian
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = sqrtm(inner)

    F = np.real(np.trace(sqrt_inner))**2

    # Numerical guard
    return min(max(F, 0.0), 1.0)


def cost_qhq(angles, rho_in, rho_target):
    """
    Cost = 1 - Uhlmann fidelity
    """

    U = U_total(*angles)

    rho_out = U @ rho_in @ U.conj().T

    F = uhlmann_fidelity(rho_target, rho_out)

    return 1 - F
