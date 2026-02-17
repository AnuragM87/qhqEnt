import numpy as np
from .qhqstates import U_total

def cost_qhq(angles, psi_in, psi_target):
    alpha1, beta1, gamma1 ,alpha2, beta2, gamma2 = angles

    psi_out = U_total(alpha1, beta1, gamma1, alpha2, beta2, gamma2) @ psi_in

    # Normalize states
    psi_out = psi_out / np.linalg.norm(psi_out)
    psi_target = psi_target / np.linalg.norm(psi_target)

    # Phase-invariant fidelity
    fidelity = np.abs(np.vdot(psi_target, psi_out))**2

    return 1 - fidelity
