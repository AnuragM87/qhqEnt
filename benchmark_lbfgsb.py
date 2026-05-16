"""
Benchmark L-BFGS-B with 50 restarts
"""
import numpy as np
import time
from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.fidelity import uhlmann_fidelity
from src.qhqstates import U_total
from src.noise_compensation import eigenvalue_filter
from src.optimizer import find_qhq_angles_multistart

rho_raw = load_density_matrix("tomography.txt")
rho_target = rho_phi_plus()
rho_filtered = eigenvalue_filter(rho_raw, rank=1)

print("=== L-BFGS-B with 50 restarts ===\n")

# Raw
t0 = time.time()
angles, fid, _ = find_qhq_angles_multistart(rho_raw, rho_target, n_starts=50)
t1 = time.time()
U = U_total(*angles)
rho_corr = U @ rho_raw @ U.conj().T
fid_corr = uhlmann_fidelity(rho_target, rho_corr)
print(f"Raw:      F = {fid_corr:.6f}  ({t1-t0:.1f}s)")

# Filtered
t0 = time.time()
angles, fid, _ = find_qhq_angles_multistart(rho_filtered, rho_target, n_starts=50)
t1 = time.time()
U = U_total(*angles)
rho_corr = U @ rho_filtered @ U.conj().T
fid_corr = uhlmann_fidelity(rho_target, rho_corr)
print(f"Filtered: F = {fid_corr:.6f}  ({t1-t0:.1f}s)")

print("\nDONE")
