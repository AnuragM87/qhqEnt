import numpy as np
from scipy.optimize import minimize, differential_evolution
from types import SimpleNamespace
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

def wrap_angles(angles):
    return np.mod(angles, 180)

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
        polish=True,
        disp=False
    )

    angles = wrap_angles(result.x)
    fidelity = 1 - result.fun

    return angles, fidelity, result

def find_qhq_angles_de_robust(rho_in, rho_target, runs=5, target_fidelity=0.98):
    """
    Production-grade DE optimizer.

    - Runs DE multiple times internally with different seeds
    - Uses 'best1exp' + larger population for thorough exploration
    - Polishes with Powell for sub-degree precision
    - Stops early if target fidelity is reached

    Usage:
        angles, fidelity, result = find_qhq_angles_de_robust(rho_in, rho_target)
    """

    bounds = [(0, 180)] * 6
    best_fidelity = -1
    best_angles = None
    best_result = None

    for i in range(runs):
        result = differential_evolution(
            cost_qhq,
            bounds=bounds,
            args=(rho_in, rho_target),
            strategy="best1exp",
            maxiter=500,
            popsize=25,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,
            seed=42 + i,
            disp=False
        )

        res_polish = minimize(
            cost_qhq,
            x0=result.x,
            args=(rho_in, rho_target),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 3000, "xtol": 1e-8, "ftol": 1e-8}
        )

        angles = wrap_angles(res_polish.x)
        fidelity = 1 - res_polish.fun

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res_polish

        if best_fidelity >= target_fidelity:
            break

    return best_angles, best_fidelity, best_result

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

def find_qhq_angles_hybrid(rho_in, rho_target):
    """
    Global search + fine tuning.
    Recommended for lab calibration.
    """

    angles, fidelity, _ = find_qhq_angles_de(rho_in, rho_target)

    angles, fidelity, res = find_qhq_angles_powell(
        rho_in, rho_target, x0=angles
    )

    return angles, fidelity, res

def find_qhq_angles_cobyla(rho_in, rho_target, x0=None):
    """
    COBYLA — derivative-free, trust-region based.
    Good for noisy fidelity landscapes.
    """
    if x0 is None:
        x0 = np.random.uniform(0, 180, 6)

    constraints = []
    for i in range(6):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i]})
        constraints.append({"type": "ineq", "fun": lambda x, i=i: 180 - x[i]})

    res = minimize(
        cost_qhq,
        x0=x0,
        args=(rho_in, rho_target),
        method="COBYLA",
        constraints=constraints,
        options={"maxiter": 3000, "rhobeg": 30, "catol": 1e-6}
    )

    angles = wrap_angles(res.x)
    fidelity = 1 - res.fun
    return angles, fidelity, res


def find_qhq_angles_spsa(rho_in, rho_target, x0=None,
                         max_iter=500,
                         c=2.0,
                         alpha=0.602, gamma=0.101,
                         A_frac=0.05,
                         momentum=0.9,
                         desired_first_step=8.0,
                         n_calibrate=5,
                         polish=True,
                         seed=None):
    """
    SPSA — Simultaneous Perturbation Stochastic Approximation (improved).

    Key improvements over vanilla SPSA:
    1. Auto-calibrated step size `a`: estimates the gradient magnitude at x0
       so the very first step moves angles by `desired_first_step` degrees.
    2. Adam-style momentum on the gradient estimate (EMA with beta=momentum)
       to suppress stochastic noise and straighten the descent path.
    3. Plateau restarts: if cost hasn't improved for 80 consecutive steps,
       restart from the best point found so far with a fresh perturbation.
    4. Powell polish: after SPSA finishes, run Powell from the best SPSA
       point to nail the local minimum precisely.

    Parameters
    ----------
    max_iter         : SPSA gradient steps
    c                : initial finite-difference radius (degrees)
    alpha            : step-size decay exponent  (Spall: 0.602)
    gamma            : perturbation decay exponent (Spall: 0.101)
    A_frac           : stability constant A = A_frac * max_iter
    momentum         : EMA coefficient for gradient smoothing (0 = no momentum)
    desired_first_step : target angle change (deg) for the first SPSA step
    n_calibrate      : gradient samples used for step-size calibration
    polish           : if True, run Powell from the best SPSA point at the end
    seed             : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = 6
    bounds_lo = np.zeros(n)
    bounds_hi = np.full(n, 180.0)
    bounds = list(zip(bounds_lo, bounds_hi))

    if x0 is None:
        x0 = rng.uniform(0, 180, n)

    theta = np.array(x0, dtype=float)
    A = A_frac * max_iter
    c_k0 = c  # initial perturbation radius

    # ── Step 1: Auto-calibrate `a` ──────────────────────────────────────────
    # Estimate the average gradient magnitude at x0 using n_calibrate samples
    g_norms = []
    for _ in range(n_calibrate):
        delta = rng.choice([-1.0, 1.0], size=n)
        tp = np.clip(theta + c_k0 * delta, bounds_lo, bounds_hi)
        tm = np.clip(theta - c_k0 * delta, bounds_lo, bounds_hi)
        g = (cost_qhq(tp, rho_in, rho_target) -
             cost_qhq(tm, rho_in, rho_target)) / (2 * c_k0 * delta)
        g_norms.append(np.linalg.norm(g))

    g_mean = np.mean(g_norms) if g_norms else 1.0
    # a_0 * g_mean / (A+1)^alpha ≈ desired_first_step  →  solve for a
    a = desired_first_step * (A + 1) ** alpha / max(g_mean, 1e-8)
    nfev = 2 * n_calibrate

    # ── Step 2: SPSA with momentum ───────────────────────────────────────────
    best_angles = theta.copy()
    best_cost = cost_qhq(theta, rho_in, rho_target)
    nfev += 1
    cost_history = [1 - best_cost]

    m_vec = np.zeros(n)          # momentum accumulator
    stall = 0                    # steps without improvement
    stall_limit = 80

    for k in range(max_iter):
        a_k = a / (A + k + 1) ** alpha
        c_k = c / (k + 1) ** gamma

        delta = rng.choice([-1.0, 1.0], size=n)
        tp = np.clip(theta + c_k * delta, bounds_lo, bounds_hi)
        tm = np.clip(theta - c_k * delta, bounds_lo, bounds_hi)

        c_plus  = cost_qhq(tp, rho_in, rho_target)
        c_minus = cost_qhq(tm, rho_in, rho_target)
        nfev += 2

        g_hat = (c_plus - c_minus) / (2 * c_k * delta)

        # Exponential moving average (momentum)
        m_vec = momentum * m_vec + (1.0 - momentum) * g_hat

        theta = np.clip(theta - a_k * m_vec, bounds_lo, bounds_hi)

        c_theta = cost_qhq(theta, rho_in, rho_target)
        nfev += 1

        if c_theta < best_cost:
            best_cost = c_theta
            best_angles = theta.copy()
            stall = 0
        else:
            stall += 1

        cost_history.append(1 - best_cost)

        # Plateau restart: jump back to best + small random nudge
        if stall >= stall_limit:
            theta = np.clip(
                best_angles + rng.uniform(-5, 5, n),
                bounds_lo, bounds_hi
            )
            m_vec = np.zeros(n)
            stall = 0

    # ── Step 3: Powell polish ────────────────────────────────────────────────
    if polish:
        res_polish = minimize(
            cost_qhq,
            x0=best_angles,
            args=(rho_in, rho_target),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 3000, "xtol": 1e-8, "ftol": 1e-8, "disp": False},
        )
        nfev += res_polish.nfev
        if res_polish.fun < best_cost:
            best_cost = res_polish.fun
            best_angles = res_polish.x

    angles = wrap_angles(best_angles)
    fidelity = 1 - best_cost

    result = SimpleNamespace(
        x=angles,
        fun=best_cost,
        nfev=nfev,
        nit=max_iter,
        success=True,
        message="SPSA + Powell completed",
        cost_history=cost_history,
    )
    return angles, fidelity, result


def find_qhq_angles_spsa_multistart(rho_in, rho_target, n_starts=5,
                                     max_iter=500, **kwargs):
    """
    Run SPSA (with auto-calibration, momentum, and polish) from n_starts
    independent random starting points. Returns the globally best result.
    """
    best_fidelity = -1
    best_angles = None
    best_result = None

    for i in range(n_starts):
        angles, fidelity, res = find_qhq_angles_spsa(
            rho_in, rho_target,
            x0=None,
            max_iter=max_iter,
            seed=i,
            **kwargs
        )
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result


# ─────────────────────────────────────────────────────────────────────────────
#  Basin-Hopping
# ─────────────────────────────────────────────────────────────────────────────

def find_qhq_angles_basin_hopping(rho_in, rho_target, x0=None,
                                   n_iter=120,
                                   T=0.05,
                                   stepsize=20.0,
                                   seed=None):
    """
    Basin-Hopping — best for smooth multi-modal bounded landscapes.

    Alternates between:
      1. Random perturbation of current position (Metropolis hop)
      2. Fast L-BFGS-B local minimization from the new point
      3. Accept/reject via Metropolis criterion (helps escape local minima)

    This is what computational chemists / quantum control researchers use
    for potential-energy surface optimization — directly analogous to the
    fidelity landscape here.

    Parameters
    ----------
    n_iter    : number of basin-hopping steps
    T         : Metropolis temperature (higher → more exploration)
    stepsize  : max angular perturbation per hop (degrees)
    seed      : RNG seed
    """
    from scipy.optimize import basinhopping

    bounds = [(0, 180)] * 6

    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, 180, 6)

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "args": (rho_in, rho_target),
        "bounds": bounds,
        "options": {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    }

    result = basinhopping(
        cost_qhq,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_iter,
        T=T,
        stepsize=stepsize,
        seed=seed,
        disp=False,
    )

    angles = wrap_angles(result.x)
    fidelity = 1 - result.fun
    return angles, fidelity, result


def find_qhq_angles_basin_hopping_multistart(rho_in, rho_target,
                                              n_starts=4, n_iter=120,
                                              **kwargs):
    """
    Basin-Hopping from n_starts independent random seeds — covers the
    full [0°, 180°]^6 space while still exploiting smooth local structure.
    """
    best_fidelity = -1
    best_angles = None
    best_result = None

    for i in range(n_starts):
        angles, fidelity, res = find_qhq_angles_basin_hopping(
            rho_in, rho_target,
            x0=None, n_iter=n_iter, seed=i, **kwargs
        )
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result


# ─────────────────────────────────────────────────────────────────────────────
#  Dual Annealing
# ─────────────────────────────────────────────────────────────────────────────

def find_qhq_angles_dual_annealing(rho_in, rho_target,
                                    maxiter=1000,
                                    initial_temp=5230.0,
                                    restart_temp_ratio=2e-5,
                                    visit=2.62,
                                    accept=-5.0,
                                    seed=None):
    """
    Dual Annealing (scipy) — combines generalized simulated annealing
    (CSA/FSA) with a fast local search phase (L-BFGS-B).

    Superior to classical SA because it uses a visiting distribution with
    heavy tails (controlled by `visit`) to escape deep local minima, then
    polishes with gradient descent.  No extra dependencies.

    Parameters
    ----------
    maxiter           : total function evaluations budget
    initial_temp      : starting temperature (higher → more exploration)
    restart_temp_ratio: ratio at which local search restarts
    visit             : visiting distribution parameter (1 < v < 3)
    accept            : acceptance distribution parameter (negative)
    seed              : RNG seed
    """
    from scipy.optimize import dual_annealing

    bounds = [(0, 180)] * 6

    result = dual_annealing(
        cost_qhq,
        bounds=bounds,
        args=(rho_in, rho_target),
        maxiter=maxiter,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        seed=seed,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "options": {"maxiter": 500, "ftol": 1e-10},
        },
        no_local_search=False,
    )

    angles = wrap_angles(result.x)
    fidelity = 1 - result.fun
    return angles, fidelity, result


# ─────────────────────────────────────────────────────────────────────────────
#  CMA-ES  (Covariance Matrix Adaptation Evolution Strategy)
# ─────────────────────────────────────────────────────────────────────────────

def find_qhq_angles_cmaes(rho_in, rho_target, x0=None,
                           sigma0=30.0,
                           maxiter=500,
                           popsize=None,
                           tol=1e-9,
                           polish=True,
                           seed=None):
    """
    CMA-ES — Covariance Matrix Adaptation Evolution Strategy.

    The gold standard for derivative-free optimization in n < 100 dimensions.
    Self-adapts both step sizes AND parameter correlations at every generation,
    making it far more efficient than plain DE or random restarts on smooth
    multi-modal landscapes.

    Particularly strong when:
      • The fidelity landscape has ridge-like or ellipsoidal basins
      • Parameters are correlated (waveplate angles interact non-linearly)
      • You want reliable convergence without manual tuning

    Requires: pip install cma  (cma >= 3.x)

    Parameters
    ----------
    sigma0   : initial step-size standard deviation (degrees).
               ~30° is a good start for a [0°,180°] search space.
    maxiter  : max CMA-ES generations
    popsize  : population per generation (None → CMA default: 4+3*ln(n))
    tol      : convergence tolerance on function value
    polish   : Powell polish of the final best point
    seed     : RNG seed
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "CMA-ES requires the 'cma' package. "
            "Install it with:  pip install cma"
        )

    n = 6
    bounds_lo = np.zeros(n)
    bounds_hi = np.full(n, 180.0)

    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(10, 170, n)   # avoid boundary edges as start

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["bounds"] = [bounds_lo.tolist(), bounds_hi.tolist()]
    opts["maxiter"] = maxiter
    opts["tolx"] = tol
    opts["tolfun"] = tol
    opts["verbose"] = -9          # completely silent
    opts["seed"] = int(seed) if seed is not None else 42
    if popsize is not None:
        opts["popsize"] = popsize

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [cost_qhq(np.array(s), rho_in, rho_target)
                     for s in solutions]
        es.tell(solutions, fitnesses)

    best_x = np.array(es.result.xbest)
    best_cost = es.result.fbest

    # Optional Powell polish
    if polish:
        bounds = list(zip(bounds_lo, bounds_hi))
        res_p = minimize(
            cost_qhq,
            x0=best_x,
            args=(rho_in, rho_target),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 3000, "xtol": 1e-9, "ftol": 1e-9},
        )
        if res_p.fun < best_cost:
            best_cost = res_p.fun
            best_x = res_p.x

    angles = wrap_angles(best_x)
    fidelity = 1 - best_cost

    result = SimpleNamespace(
        x=angles,
        fun=best_cost,
        nfev=es.result.evaluations,
        nit=es.result.iterations,
        success=True,
        message="CMA-ES + Powell completed",
        cma_result=es.result,
    )
    return angles, fidelity, result


def find_qhq_angles_cmaes_multistart(rho_in, rho_target, n_starts=3,
                                      sigma0=30.0, maxiter=500, **kwargs):
    """
    CMA-ES from n_starts independent random seeds.
    Even 2–3 restarts dramatically reduces the chance of a bad local minimum.
    """
    best_fidelity = -1
    best_angles = None
    best_result = None

    for i in range(n_starts):
        angles, fidelity, res = find_qhq_angles_cmaes(
            rho_in, rho_target,
            x0=None, sigma0=sigma0, maxiter=maxiter, seed=i, **kwargs
        )
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_angles = angles
            best_result = res

    return best_angles, best_fidelity, best_result
