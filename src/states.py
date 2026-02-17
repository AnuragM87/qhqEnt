import numpy as np
H = np.array([1, 0], dtype=complex)   # Horizontal
V = np.array([0, 1], dtype=complex)   # Vertical

D = np.array([1/np.sqrt(2),  1/np.sqrt(2)], dtype=complex)   # Diagonal
A = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)   # Anti-diagonal

R = np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex)  # Right circular
L = np.array([1/np.sqrt(2),  1j/np.sqrt(2)], dtype=complex)  # Left circular

#Tensor product
def tp(a, b):
    """Tensor product of two single-photon states"""
    return np.kron(a, b)


# Basis states for two photons
HH = tp(H, H)
HV = tp(H, V)
VH = tp(V, H)
VV = tp(V, V)

# Bell State

def phi_plus():
    """|Φ+⟩ = (|HH⟩ + |VV⟩)/√2"""
    return (HH + VV) / np.sqrt(2)

def phi_minus():
    """|Φ-⟩ = (|HH⟩ - |VV⟩)/√2"""
    return (HH - VV) / np.sqrt(2)

def psi_plus():
    """|Ψ+⟩ = (|HV⟩ + |VH⟩)/√2"""
    return (HV + VH) / np.sqrt(2)

def psi_minus():
    """|Ψ-⟩ = (|HV⟩ - |VH⟩)/√2"""
    return (HV - VH) / np.sqrt(2)


# def drifted_phi_plus(epsilon=0.05, phase=0.1, leakage=0.02):
#     """
#     Generate a realistic imperfect Φ+ state.

#     Parameters
#     ----------
#     epsilon : float
#         Amplitude imbalance (0 → perfect balance)
#     phase : float
#         Relative phase error between HH and VV
#     leakage : float
#         Small HV/VH mixing due to imperfections

#     Returns
#     -------
#     psi : np.ndarray (4,)
#         Normalized two-photon state vector
#     """

#     alpha = np.cos(np.pi/4 + epsilon)
#     beta  = np.sin(np.pi/4 + epsilon) * np.exp(1j * phase)

#     psi = alpha * HH + beta * VV + leakage * (HV + VH)
#     psi /= np.linalg.norm(psi)

#     return psi


# # ============================================================
# # 6️⃣ Density matrix utilities
# # ============================================================

# def density_matrix(psi):
#     """
#     Convert state vector → density matrix.
#     """
#     return np.outer(psi, psi.conj())


# def fidelity(rho, target_state):
#     """
#     Compute fidelity between density matrix and target pure state.
#     """
#     return np.real(target_state.conj().T @ rho @ target_state)


# # ==========================================================