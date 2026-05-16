import numpy as np

def UQ(theta_deg):
    th = np.deg2rad(theta_deg)
    c = np.cos(th)
    s = np.sin(th)
    c2 = c * c
    s2 = s * s
    off = s * c * (1 - 1j)

    return np.array([
        [c2 + 1j * s2, off],
        [off,         s2 + 1j * c2]
    ], dtype=complex)

def UH(theta_deg):
    th = np.deg2rad(theta_deg)
    c2 = np.cos(2 * th)
    s2 = np.sin(2 * th)

    return np.array([
        [c2,  s2],
        [s2, -c2]
    ], dtype=complex)
