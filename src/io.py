import numpy as np

def load_density_matrix(filepath):
    data = np.loadtxt(filepath, dtype=complex)
    rho = data.reshape((4, 4))

    # enforce Hermitian & normalization
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    return rho
