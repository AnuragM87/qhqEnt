import numpy as np
import matplotlib.pyplot as plt

def showHisto(rho):
    probs = np.real(np.diag(rho))
    labels = ['HH', 'HV', 'VH', 'VV']

    plt.figure()
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Population (Tomography)")
    plt.show()
