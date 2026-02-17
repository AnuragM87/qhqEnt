import numpy as np

def load_density_matrix(path):
    return np.loadtxt(path, delimiter=',', dtype=complex)
