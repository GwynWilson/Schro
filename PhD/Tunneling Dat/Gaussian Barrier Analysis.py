import numpy as np
import matplotlib.pyplot as plt


def plotReg(scale=""):
    plt.figure()
    plt.plot(V0, sim, label="Sim")
    plt.plot(V0, imp, linestyle="--")
    plt.legend()
    if scale != "":
        plt.yscale(scale)
    plt.xlabel("v0")
    plt.ylabel("Transpmisson Probability")
    plt.savefig("V0_Transmission_Gauss_{}".format(scale))
    plt.show()


def plotDiff():
    plt.plot(V0, abs(np.asarray(sim) - np.asarray(imp)))
    plt.tight_layout(pad=2)
    plt.title("Difference In Predicted and Actual Tunneling")
    plt.xlabel("V0")
    plt.ylabel("Absolute Value of Difference")
    plt.savefig("V0_Transmission_Difference_Gauss")
    plt.show()

N, dx, omeg, dt, k0 = np.loadtxt("Gauss_Barrier_var.txt")
dat = np.loadtxt("Gauss_Barrier.txt")
V0, sim, imp = [list(col) for col in zip(*dat)]

plotReg()
plotDiff()