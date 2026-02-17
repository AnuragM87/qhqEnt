import matplotlib.pyplot as plt
import numpy as np

def showHisto(psi_out):
    labels = ["HH", "HV", "VH", "VV"]
    probabilities = np.abs(psi_out)**2

    plt.bar(labels, probabilities)
    plt.title("Output State Probabilities")
    plt.xlabel("Basis State")
    plt.ylabel("Probability")
    plt.show()