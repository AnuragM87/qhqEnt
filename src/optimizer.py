from scipy.optimize import minimize
import numpy as np
from .cost import cost_qhq

def find_qhq_angles(rho_in, rho_target, x0=None):
    if x0 is None:
        x0 = np.zeros(6)

    bounds = [(0, 180)] * 6

    res = minimize(
        cost_qhq,
        x0=x0,
        args=(rho_in, rho_target),
        method="L-BFGS-B",
        bounds=bounds
    )

    return res.x, 1 - res.fun, res

def find_qhq_angles_multistart(rho_in, rho_target, n_starts=10):
    best_F = -1
    best_angles = None
    best_res = None

    for _ in range(n_starts):
        x0 = np.random.uniform(0, 180, size=6)
        angles, F, res = find_qhq_angles(rho_in, rho_target, x0)

        if F > best_F:
            best_F = F
            best_angles = angles
            best_res = res

    return best_angles, best_F, best_res
