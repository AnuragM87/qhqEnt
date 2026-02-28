import numpy as np

def rho_phi_plus():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())

def rho_phi_minus():
    psi = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())

def rho_psi_minus():
    psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())
