
import numpy as np
from .waveplates import UH , UQ

def UqhqArm1(alpha1,beta1,gamma1):
    return UQ(alpha1) @ UH(beta1) @ UQ(gamma1)

def UqhqArm2(alpha2,beta2,gamma2):
    return UQ(alpha2) @ UH(beta2) @ UQ(gamma2)

def U_total(alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    U1 = UqhqArm1(alpha1, beta1, gamma1)
    U2 = UqhqArm2(alpha2, beta2, gamma2)
    return np.kron(U1, U2)