import numpy as np
import matplotlib.pyplot as plt


def plotReg(scale=""):
    plt.figure()
    plt.plot(V0, sim, label="Sim")
    plt.plot(V0, theory, label="Theory", linestyle="--")
    plt.plot(V0, impedence, label="Impedence", linestyle="--")
    plt.legend()
    if scale != "":
        plt.yscale(scale)
    plt.xlabel("v0")
    plt.ylabel("Transpmisson Probability")
    plt.savefig("V0_Transmission_{}".format(scale))
    plt.show()


def plotDiff():
    plt.plot(V0, abs(np.asarray(sim) - np.asarray(impedence)))
    plt.tight_layout(pad=2)
    plt.title("Difference In Predicted and Actual Tunneling")
    plt.xlabel("V0")
    plt.ylabel("Absolute Value of Difference")
    plt.savefig("V0_Transmission_Difference")
    plt.show()


N, dx, L, dt, k0 = np.loadtxt("Square_Barrier_var.txt")
dat = np.loadtxt("Square_Barrier.txt")
V0, sim, theory, impedence = [list(col) for col in zip(*dat)]

plotReg()
plotReg("log")
plotDiff()
