import numpy as np

H = np.array([1, 0], dtype=complex)
V = np.array([0, 1], dtype=complex)

def kron(a, b):
    return np.kron(a, b)

def density(psi):
    return np.outer(psi, psi.conj())

# Bell states (density matrices)

def rho_phi_plus():
    psi = (kron(H, H) + kron(V, V)) / np.sqrt(2)
    return density(psi)

def rho_phi_minus():
    psi = (kron(H, H) - kron(V, V)) / np.sqrt(2)
    return density(psi)

def rho_psi_plus():
    psi = (kron(H, V) + kron(V, H)) / np.sqrt(2)
    return density(psi)

def rho_psi_minus():
    psi = (kron(H, V) - kron(V, H)) / np.sqrt(2)
    return density(psi)
