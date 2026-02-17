import numpy as np
from src.states import phi_minus ,phi_plus ,psi_minus
from src.qhqstates import U_total
from src.optimizer import  find_qhq_angles_multistart
from plot import showHisto

psi_in=psi_minus()
print("Input state:", psi_in)
target_state = phi_plus()

angles, fidelity ,best_result= find_qhq_angles_multistart(psi_in, target_state)
print(angles, fidelity)
psi_out = U_total(*angles) @ psi_in
rho = np.outer(psi_out, psi_out.conj())
print(rho)
# showHisto(psi_out)

# def remove_global_phase(psi):
#     return psi * np.exp(-1j * np.angle(psi[0]))

# psi_out_nophase = remove_global_phase(psi_out)
# print("Output state (no global phase):", psi_out_nophase)


# showHisto(psi_out_nophase)