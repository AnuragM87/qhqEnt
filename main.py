import numpy as np
from src.load_density import load_density_matrix
from src.states import rho_phi_plus
from src.optimizer import find_qhq_angles_multistart
from src.qhqstates import U_total
from plot import showHisto

# -----------------------------
# Load tomography data
# -----------------------------
rho_in = load_density_matrix("data/tomography.txt")
print("Loaded tomography matrix:\n", rho_in)

showHisto(rho_in)

# -----------------------------
# Target: ideal or reconstructed Bell state
# -----------------------------
rho_target = rho_phi_plus()

print("\nInitial fidelity:",
      np.real(np.trace(rho_target @ rho_in)))

# -----------------------------
# Optimize QHQ angles
# -----------------------------
angles, fidelity, _ = find_qhq_angles_multistart(rho_in, rho_target)

print("\nBest angles:", angles)
print("Final fidelity:", fidelity)

# -----------------------------
# Apply correction
# -----------------------------
U = U_total(*angles)
rho_out = U @ rho_in @ U.conj().T

print("\nCorrected density matrix:\n", rho_out)

showHisto(rho_out)
