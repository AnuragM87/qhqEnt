from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.optimizer import find_qhq_angles_cobyla, find_qhq_angles_multistart,find_qhq_angles_hybrid,find_qhq_angles_de,find_qhq_angles_powell_multistart,find_qhq_angles_de_multirun
from src.qhqstates import U_total
from src.fidelity import uhlmann_fidelity
from plot import showHisto
import numpy as np

# Load experimental tomography
rho_in = load_density_matrix("tomography.txt")

# Target Bell state
rho_target = rho_phi_plus()
initial_fidelity=uhlmann_fidelity(rho_target, rho_in)
# showHisto(rho_in)

# Optimize waveplate angles
MAX_ITERS = 10
TARGET = 0.98

best_fidelity = initial_fidelity
best_angles = None

for _ in range(MAX_ITERS):
    angles, fidelity, _ = find_qhq_angles_cobyla(rho_in, rho_target)
    print("Iteration...")
    if fidelity > best_fidelity:
        best_fidelity = fidelity
        best_angles = angles

    if best_fidelity >= TARGET:
        break


print("Optimal angles:", best_angles)
# print("Fidelity:", fidelity)

# Apply correction
U = U_total(*best_angles)
rho_out = U @ rho_in @ U.conj().T
# print("Corrected density matrix:\n", rho_out)
# Plot results
# showHisto(rho_in)
# showHisto(rho_out)

print("Fidelity before:", uhlmann_fidelity(rho_target, rho_in))
print("Fidelity after :", uhlmann_fidelity(rho_target, rho_out))
# print("Imag part before:\n", np.imag(rho_in))
# print("Imag part after:\n", np.imag(rho_out))
