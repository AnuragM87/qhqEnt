import numpy as np
from .qhqstates import U_total
from .fidelity import uhlmann_fidelity

def cost_qhq(angles, rho_in, rho_target):
    U = U_total(*angles)
    rho_out = U @ rho_in @ U.conj().T
    F = uhlmann_fidelity(rho_target, rho_out)
    return 1 - F
