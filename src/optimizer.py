import numpy as np
from scipy.optimize import minimize
from .cost import cost_qhq

def find_qhq_angles(rho_in, rho_target, x0=None):
    if x0 is None:
        x0 = np.random.uniform(0, 180, 6)

    bounds = [(0, 180)] * 6

    res = minimize(
        cost_qhq,
        x0=x0,
        args=(rho_in, rho_target),
        method="L-BFGS-B",
        bounds=bounds
    )

    return res.x, 1 - res.fun, res


def find_qhq_angles_multistart(rho_in, rho_target, n_starts=8):
    best_fidelity = -1
    best_angles = None
    best_result = None

    for _ in range(n_starts):
        angles, fidelity, res = find_qhq_angles(rho_in, rho_target)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result
