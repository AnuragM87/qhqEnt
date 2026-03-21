# # # import numpy as np
# # # from scipy.linalg import sqrtm

# # # def uhlmann_fidelity(rho, sigma):
# # #     rho = (rho + rho.conj().T) / 2
# # #     sigma = (sigma + sigma.conj().T) / 2

# # #     sqrt_rho = sqrtm(rho)
# # #     inner = sqrt_rho @ sigma @ sqrt_rho
# # #     sqrt_inner = sqrtm(inner)

# # #     F = np.real(np.trace(sqrt_inner))**2
# # #     return min(max(F, 0.0), 1.0)


# # import numpy as np
# # from scipy.linalg import sqrtm

# # def uhlmann_fidelity(rho, sigma, eps=1e-10):
# #     # Make matrices Hermitian
# #     rho = (rho + rho.conj().T) / 2
# #     sigma = (sigma + sigma.conj().T) / 2

# #     # Eigenvalue cleanup
# #     def clean(mat):
# #         vals, vecs = np.linalg.eigh(mat)
# #         vals[vals < eps] = 0
# #         return vecs @ np.diag(vals) @ vecs.conj().T

# #     rho = clean(rho)
# #     sigma = clean(sigma)

# #     sqrt_rho = sqrtm(rho)
# #     inner = sqrt_rho @ sigma @ sqrt_rho
# #     sqrt_inner = sqrtm(inner)

# #     return np.real(np.trace(sqrt_inner))**2

# import numpy as np
# from scipy.linalg import sqrtm

# def uhlmann_fidelity(rho, sigma, eps=1e-10):
#     # --- Ensure Hermitian ---
#     rho = (rho + rho.conj().T) / 2
#     sigma = (sigma + sigma.conj().T) / 2

#     # --- Normalize trace ---
#     rho = rho / np.trace(rho)
#     sigma = sigma / np.trace(sigma)

#     # --- Remove negative eigenvalues ---
#     def clean(mat):
#         vals, vecs = np.linalg.eigh(mat)
#         vals[vals < eps] = 0
#         return vecs @ np.diag(vals) @ vecs.conj().T

#     rho = clean(rho)
#     sigma = clean(sigma)

#     # --- Uhlmann fidelity ---
#     sqrt_rho = sqrtm(rho)
#     inner = sqrt_rho @ sigma @ sqrt_rho
#     sqrt_inner = sqrtm(inner)

#     F = np.real(np.trace(sqrt_inner))**2

#     # --- Clamp to physical range ---
#     return float(np.clip(F, 0.0, 1.0))

import numpy as np

def uhlmann_fidelity(rho, sigma, eps=1e-12):
    # Ensure Hermitian
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    # Normalize trace
    rho = rho / np.trace(rho)
    sigma = sigma / np.trace(sigma)

    # Remove tiny negatives from tomography noise
    def clean(mat):
        vals, vecs = np.linalg.eigh(mat)
        vals[vals < eps] = 0
        return vecs @ np.diag(vals) @ vecs.conj().T

    rho = clean(rho)
    sigma = clean(sigma)

    # --- Eigenvalue method ---
    eigvals = np.linalg.eigvals(rho @ sigma)

    # Numerical cleanup
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0

    F = (np.sum(np.sqrt(eigvals)))**2
    return float(np.clip(F, 0.0, 1.0))



    # import numpy as np

# def pure_state_fidelity(rho_exp, rho_target):
#     """
#     Fidelity between an experimental state and a target PURE state.

#     Parameters
#     ----------
#     rho_exp : density matrix (NxN)
#     rho_target : pure state density matrix (NxN)

#     Returns
#     -------
#     Fidelity (float)
#     """

#     # Ensure Hermitian
#     rho_exp = (rho_exp + rho_exp.conj().T) / 2
#     rho_target = (rho_target + rho_target.conj().T) / 2

#     # Normalize traces
#     rho_exp = rho_exp / np.trace(rho_exp)
#     rho_target = rho_target / np.trace(rho_target)

#     # Fidelity: Tr(rho_target * rho_exp)
#     F = np.trace(rho_target @ rho_exp)

#     return float(np.real(F))