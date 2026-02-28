from src.io import load_density_matrix
from src.states import rho_phi_plus
from src.optimizer import find_qhq_angles_multistart
from src.qhqstates import U_total
from src.fidelity import uhlmann_fidelity
from plot import showHisto
import numpy as np

# Load experimental tomography
rho_in = load_density_matrix("tomography.txt")

# Target Bell state
rho_target = rho_phi_plus()
showHisto(rho_in)

# Optimize waveplate angles
fidelity=0
counter=0
while fidelity<0.98 and counter<20:
    angles, fidelity, _ = find_qhq_angles_multistart(rho_in, rho_target)
    print("iteration...")
    counter+=1


print("Optimal angles:", angles)
print("Fidelity:", fidelity)

# Apply correction
U = U_total(*angles)
rho_out = U @ rho_in @ U.conj().T
# print("Corrected density matrix:\n", rho_out)
# Plot results
# showHisto(rho_in)
showHisto(rho_out)

# print("Fidelity before:", uhlmann_fidelity(rho_target, rho_in))
# print("Fidelity after :", uhlmann_fidelity(rho_target, rho_out))
# print("Imag part before:\n", np.imag(rho_in))
# print("Imag part after:\n", np.imag(rho_out))
