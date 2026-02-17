from scipy.optimize import minimize
from .cost import cost_qhq
import numpy as np


def find_qhq_angles(psi_in, psi_target, x0=(0, 0, 0,0,0,0)):
    bounds = [(0, 180), (0, 180), (0, 180),(0, 180), (0, 180), (0, 180)]

    res = minimize(
        cost_qhq,
        x0=x0,
        args=(psi_in, psi_target),
        method="L-BFGS-B",
        bounds=bounds
    )

    return res.x, 1 - res.fun, res

# def find_qhq_angles_multistart(psi_in, psi_target, n_starts=10):
#     best_fidelity = -1
#     best_angles = None

#     for _ in range(n_starts):
#         x0 = np.random.uniform(0, 180, size=3)
#         angles, fidelity, _ = find_qhq_angles(psi_in, psi_target, x0)

#         if fidelity > best_fidelity:
#             best_fidelity = fidelity
#             best_angles = angles

#     return best_angles, best_fidelity

def find_qhq_angles_multistart(psi_in, psi_target, n_starts=10):
    best_fidelity = -1
    best_angles = None
    best_result = None

    for _ in range(n_starts):
        x0 = np.random.uniform(0, 180, size=6)

        angles, fidelity, res = find_qhq_angles(
            psi_in, psi_target, x0
        )

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result

