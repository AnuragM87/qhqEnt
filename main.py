from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.optimizer import find_qhq_angles_cobyla, find_qhq_angles_multistart,find_qhq_angles_hybrid,find_qhq_angles_de,find_qhq_angles_powell_multistart,find_qhq_angles_de_multirun
from src.qhqstates import U_total
from src.fidelity import uhlmann_fidelity
from src.noise_compensation import depolarization_compensation, eigenvalue_filter
from plot import showHisto
import numpy as np

# Load experimental tomography
rho_raw = load_density_matrix("tomography.txt")

# Target Bell state
rho_target = rho_phi_plus()

# =========================================================
# PREPROCESSING: Pick ONE method (comment/uncomment)
# =========================================================

# Method 1: No cleaning (use raw tomography data)
# rho_in = rho_raw

# Method 2: Depolarization compensation (inverts depolarizing noise)
# rho_in, p_est = depolarization_compensation(rho_raw)

# Method 3: Eigenvalue filtering (keeps top eigenstate, discards noise)
rho_in = eigenvalue_filter(rho_raw, rank=1)

# =========================================================

initial_fidelity=uhlmann_fidelity(rho_target, rho_in)
print(f"Initial fidelity: {initial_fidelity:.4f}")
# showHisto(rho_in)

# Optimize waveplate angles
MAX_ITERS = 20
TARGET = 0.98

best_fidelity = initial_fidelity
best_angles = None

for _ in range(MAX_ITERS):
    angles, fidelity, _ = find_qhq_angles_multistart(rho_in, rho_target)
    print("Iteration...")
    if fidelity > best_fidelity:
        best_fidelity = fidelity
        best_angles = angles

    if best_fidelity >= TARGET:
        break


print("Optimal angles:", best_angles)
print("Fidelity:", fidelity)

# Apply correction
U = U_total(*best_angles)
rho_out = U @ rho_in @ U.conj().T
print("Corrected density matrix:\n", rho_out)
# Plot results
showHisto(rho_raw)
# showHisto(rho_in)
showHisto(rho_out)

print("Fidelity before:", uhlmann_fidelity(rho_target, rho_in))
print("Fidelity after :", uhlmann_fidelity(rho_target, rho_out))
print("Imag part before:\n", np.imag(rho_in))
print("Imag part after:\n", np.imag(rho_out))
