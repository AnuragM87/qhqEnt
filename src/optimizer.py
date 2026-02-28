import numpy as np
from scipy.optimize import minimize, differential_evolution
from .cost import cost_qhq   # your existing cost function



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


# import numpy as np
# from scipy.optimize import minimize
# from .cost import cost_qhq

# def find_qhq_angles(rho_in, rho_target, x0=None):
#     if x0 is None:
#         x0 = np.random.uniform(0, 180, 6)

#     bounds = [(0, 180)] * 6  # Powell supports bounds in SciPy

#     res = minimize(
#         cost_qhq,
#         x0=x0,
#         args=(rho_in, rho_target),
#         method="Powell",
#         bounds=bounds,
#         options={
#             "maxiter": 2000,
#             "xtol": 1e-6,
#             "ftol": 1e-6,
#             "disp": False
#         }
#     )

#     return res.x, 1 - res.fun, res


# def find_qhq_angles_multistart(rho_in, rho_target, n_starts=8):
#     best_fidelity = -1
#     best_angles = None
#     best_result = None

#     for _ in range(n_starts):
#         angles, fidelity, res = find_qhq_angles(rho_in, rho_target)

#         if fidelity > best_fidelity:
#             best_fidelity = fidelity
#             best_angles = angles
#             best_result = res

#     return best_angles, best_fidelity, best_result



# =========================================================
# Utility: wrap angles to 0–180 degrees
# =========================================================
def wrap_angles(angles):
    return np.mod(angles, 180)


# =========================================================
# 1. POWELL OPTIMIZER
# =========================================================
def find_qhq_angles_powell(rho_in, rho_target, x0=None):
    """
    Local optimizer (good for fine tuning).
    """

    if x0 is None:
        x0 = np.random.uniform(0, 180, 6)

    bounds = [(0, 180)] * 6

    res = minimize(
        cost_qhq,
        x0=x0,
        args=(rho_in, rho_target),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "xtol": 1e-6,
            "ftol": 1e-6,
            "disp": False
        }
    )

    angles = wrap_angles(res.x)
    fidelity = 1 - res.fun

    return angles, fidelity, res


# =========================================================
# 2. DIFFERENTIAL EVOLUTION (GLOBAL OPTIMIZER)
# =========================================================
def find_qhq_angles_de(rho_in, rho_target):
    """
    Global optimizer — best for escaping local minima.
    """

    bounds = [(0, 180)] * 6

    result = differential_evolution(
        cost_qhq,
        bounds=bounds,
        args=(rho_in, rho_target),
        strategy="best1bin",
        maxiter=300,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,   # final local refinement
        disp=False
    )

    angles = wrap_angles(result.x)
    fidelity = 1 - result.fun

    return angles, fidelity, result


# =========================================================
# 3. MULTISTART POWELL
# =========================================================
def find_qhq_angles_powell_multistart(rho_in, rho_target, n_starts=10):
    best_fidelity = -1
    best_angles = None
    best_result = None

    for _ in range(n_starts):
        angles, fidelity, res = find_qhq_angles_powell(rho_in, rho_target)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result


# =========================================================
# 4. MULTIRUN DE (optional for noisy lab data)
# =========================================================
def find_qhq_angles_de_multirun(rho_in, rho_target, runs=3):
    best_fidelity = -1
    best_angles = None
    best_result = None

    for _ in range(runs):
        angles, fidelity, result = find_qhq_angles_de(rho_in, rho_target)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = result

    return best_angles, best_fidelity, best_result


# =========================================================
# 5. HYBRID: DE → POWELL (BEST PERFORMANCE)
# =========================================================
def find_qhq_angles_hybrid(rho_in, rho_target):
    """
    Global search + fine tuning.
    Recommended for lab calibration.
    """

    # Step 1: global search
    angles, fidelity, _ = find_qhq_angles_de(rho_in, rho_target)

    # Step 2: local refinement
    angles, fidelity, res = find_qhq_angles_powell(
        rho_in, rho_target, x0=angles
    )

    return angles, fidelity, res