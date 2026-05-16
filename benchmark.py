"""
Benchmark all optimizers: Raw vs Eigenvalue Filter
"""
import numpy as np
import json
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

# Load data
rho_raw = load_density_matrix("tomography.txt")
rho_target = rho_phi_plus()
rho_filtered = eigenvalue_filter(rho_raw, rank=1)

# Initial fidelities
init_raw = uhlmann_fidelity(rho_target, rho_raw)
init_filtered = uhlmann_fidelity(rho_target, rho_filtered)

print(f"INITIAL_RAW={init_raw:.6f}")
print(f"INITIAL_FILTERED={init_filtered:.6f}")

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

    # Raw data
    print(f"  Running with raw data...")
    t0 = time.time()
    angles_raw, fid_raw, _ = fn(rho_raw, rho_target)
    time_raw = time.time() - t0
    # Apply correction to get actual corrected fidelity
    U = U_total(*angles_raw)
    rho_corr_raw = U @ rho_raw @ U.conj().T
    fid_corr_raw = uhlmann_fidelity(rho_target, rho_corr_raw)
    print(f"  Raw: F={fid_corr_raw:.6f} ({time_raw:.1f}s)")

    # Eigenvalue filtered
    print(f"  Running with eigenvalue filter...")
    t0 = time.time()
    angles_filt, fid_filt, _ = fn(rho_filtered, rho_target)
    time_filt = time.time() - t0
    U = U_total(*angles_filt)
    rho_corr_filt = U @ rho_filtered @ U.conj().T
    fid_corr_filt = uhlmann_fidelity(rho_target, rho_corr_filt)
    print(f"  Filtered: F={fid_corr_filt:.6f} ({time_filt:.1f}s)")

    results.append({
        "name": name,
        "fid_raw": fid_corr_raw,
        "fid_filtered": fid_corr_filt,
        "time_raw": time_raw,
        "time_filtered": time_filt,
    })

# Print summary
print("\n\n========== RESULTS ==========")
print(json.dumps(results, indent=2))
print("DONE")
