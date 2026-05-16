"""
Benchmark all optimizers with Depolarization Compensation
"""
import numpy as np
import json
import time
from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.fidelity import uhlmann_fidelity
from src.qhqstates import U_total
from src.noise_compensation import depolarization_compensation
from src.optimizer import (
    find_qhq_angles_multistart,
    find_qhq_angles_powell_multistart,
    find_qhq_angles_de,
    find_qhq_angles_de_robust,
    find_qhq_angles_de_multirun,
    find_qhq_angles_hybrid,
    find_qhq_angles_cobyla,
)

rho_raw = load_density_matrix("tomography.txt")
rho_target = rho_phi_plus()
rho_depol, p_est = depolarization_compensation(rho_raw)

init_raw = uhlmann_fidelity(rho_target, rho_raw)
init_depol = uhlmann_fidelity(rho_target, rho_depol)
print(f"INITIAL_RAW={init_raw:.6f}")
print(f"INITIAL_DEPOL={init_depol:.6f}")
print(f"NOISE_P={p_est:.6f}")

optimizers = [
    ("Multistart L-BFGS-B", find_qhq_angles_multistart),
    ("Powell Multistart", find_qhq_angles_powell_multistart),
    ("Differential Evolution", find_qhq_angles_de),
    ("DE Robust", find_qhq_angles_de_robust),
    ("DE Multi-run", find_qhq_angles_de_multirun),
    ("Hybrid (DE→Powell)", find_qhq_angles_hybrid),
    ("COBYLA", find_qhq_angles_cobyla),
]

results = []
for name, fn in optimizers:
    print(f"\n--- {name} ---")
    t0 = time.time()
    angles, fid, _ = fn(rho_depol, rho_target)
    elapsed = time.time() - t0
    U = U_total(*angles)
    rho_corr = U @ rho_depol @ U.conj().T
    fid_corr = uhlmann_fidelity(rho_target, rho_corr)
    print(f"  Depol: F={fid_corr:.6f} ({elapsed:.1f}s)")
    results.append({"name": name, "fid_depol": fid_corr, "time": elapsed})

print("\n\n========== RESULTS ==========")
print(json.dumps(results, indent=2))
print("DONE")
