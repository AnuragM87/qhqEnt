"""
Benchmark all optimizers with 10 outer iterations each
"""
import numpy as np
import time
from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.fidelity import uhlmann_fidelity
from src.qhqstates import U_total
from src.noise_compensation import eigenvalue_filter
from src.optimizer import (
    find_qhq_angles_multistart,
    find_qhq_angles_powell_multistart,
    find_qhq_angles_de,
    find_qhq_angles_de_robust,
    find_qhq_angles_de_multirun,
    find_qhq_angles_hybrid,
    find_qhq_angles_cobyla,
)

OUTER_ITERS = 10
TARGET_F = 0.98

rho_raw = load_density_matrix("tomography.txt")
rho_target = rho_phi_plus()
rho_filtered = eigenvalue_filter(rho_raw, rank=1)

init_raw = uhlmann_fidelity(rho_target, rho_raw)
init_filt = uhlmann_fidelity(rho_target, rho_filtered)
print(f"Initial Raw: {init_raw:.6f}")
print(f"Initial Filtered: {init_filt:.6f}")

optimizers = [
    ("Multistart L-BFGS-B", find_qhq_angles_multistart),
    ("Powell Multistart", find_qhq_angles_powell_multistart),
    ("Differential Evolution", find_qhq_angles_de),
    ("DE Robust", find_qhq_angles_de_robust),
    ("DE Multi-run", find_qhq_angles_de_multirun),
    ("Hybrid (DE->Powell)", find_qhq_angles_hybrid),
    ("COBYLA", find_qhq_angles_cobyla),
]

def run_with_outer_loop(fn, rho_in, rho_target, outer_iters, target_f):
    best_fid = uhlmann_fidelity(rho_target, rho_in)
    best_angles = None
    for i in range(outer_iters):
        angles, fid, _ = fn(rho_in, rho_target)
        if fid > best_fid:
            best_fid = fid
            best_angles = angles
        if best_fid >= target_f:
            break
    # Apply best correction
    if best_angles is not None:
        U = U_total(*best_angles)
        rho_corr = U @ rho_in @ U.conj().T
        fid_corr = uhlmann_fidelity(rho_target, rho_corr)
    else:
        fid_corr = best_fid
    return fid_corr

print(f"\n{'='*60}")
print(f"Running each optimizer with {OUTER_ITERS} outer iterations")
print(f"{'='*60}\n")

for name, fn in optimizers:
    print(f"--- {name} ---")

    # Raw
    t0 = time.time()
    fid_raw = run_with_outer_loop(fn, rho_raw, rho_target, OUTER_ITERS, TARGET_F)
    time_raw = time.time() - t0

    # Filtered
    t0 = time.time()
    fid_filt = run_with_outer_loop(fn, rho_filtered, rho_target, OUTER_ITERS, TARGET_F)
    time_filt = time.time() - t0

    print(f"  Raw:      F={fid_raw:.6f}  ({time_raw:.1f}s)")
    print(f"  Filtered: F={fid_filt:.6f}  ({time_filt:.1f}s)")
    print()

print("DONE")
