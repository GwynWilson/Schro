import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import glob


def exp(x, a, b, c):
    return a * np.exp(b * x) + c


def pol(x, a, b):
    return a * x + b


def t_theory(L, V0, E, m=1, hbar=1):
    amp = 16 * E / V0 * (1 - (E / V0))
    # amp = 16 / (3+ (V0/E))
    # amp=1
    coeff = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return amp * np.exp(- 2 * L * coeff)


def t_theory2(L, V0, E, m=1, hbar=1):
    # print(L,V0,E)
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return (1 + 1 / 4 * (k1 / k2 + k2 / k1) ** 2 * np.sinh(k2 * L) ** 2) ** (-1)


def split_dat(dat):
    w = []
    t = []
    for i in range(len(dat)):
        w.append(dat[i][0])
        t.append(dat[i][1])
    return w, t


def square_barrier():
    v0, k, hbar, m, dx = np.loadtxt("Square_Barrier_var.txt")
    E = (hbar ** 2) * (k ** 2) / (2 * m)

    dat = np.loadtxt("Square_Barrier.txt")
    w, t = split_dat(dat)
    # dat2 = np.loadtxt("Square_Barrier_B.txt")
    # w2, t2 = split_dat(dat2)
    #
    # dat3 = np.loadtxt("Square_Barrier_h.txt")
    # w3, t3 = split_dat(dat3)

    w_gen = np.linspace(min(w), max(w), 100)

    # Curve fit
    # args = (1, -1, 1)
    # ppot, covt = curve_fit(exp, w, t, p0=args)
    # fitted = exp(w_gen, ppot[0], ppot[1], ppot[2])

    theory = [t_theory(i * dx, v0, E) for i in w_gen]
    theory2 = [t_theory2(i * dx, v0, E) for i in w_gen]

    # Log fit
    # log_t = np.log(t)
    # ppot2, covt2 = curve_fit(pol, w, log_t, p0=(-1, 1))
    # fitted2 = pol(w_gen, ppot2[0], ppot[1])

    plt.semilogy(dx * np.asarray(w), np.asarray(t), linestyle="", marker="o", label="Simulation")
    # plt.semilogy(w2, np.asarray(t2), linestyle="", marker="o", label="Simulation first")
    # plt.semilogy(w3, np.asarray(t3), linestyle="", marker="o", label="Simulation half")
    # plt.plot(w_gen, np.exp(fitted2), label="Fitted Exp")
    # plt.plot(w, t, linestyle="", marker="o", label="Simulation")
    plt.semilogy(dx * np.asarray(w_gen), theory, label="Exp")
    plt.semilogy(dx * np.asarray(w_gen), theory2, label="Theoretical")
    plt.title("Square Barrier Transmission Probability")
    plt.xlabel("Barrier Length")
    plt.ylabel("Transmission Probability Log")
    plt.legend(loc="best")
    plt.savefig("Square_Barrier_Tunneling_log")
    plt.show()

    # plt.plot(w_gen, fitted, label="Fitted Exp")
    plt.plot(dx * np.asarray(w), t, label="Simulation", linestyle="", marker="o")
    # plt.yscale("log")
    plt.plot(dx * np.asarray(w_gen), theory, label="Exp")
    plt.plot(dx * np.asarray(w_gen), theory2, label="Theoretical")
    plt.title("Square Barrier Transmission Probability")
    plt.xlabel("Barrier Length")
    plt.ylabel("Transmission Probability")
    plt.legend(loc="best")
    plt.savefig("Square_Barrier_Tunneling")
    plt.show()
    return 0


def load_gauss(g_list, log=False):
    for i in g_list:
        temp = list(i.split("_"))
        dat = np.loadtxt(i)
        w, t = split_dat(dat)
        plt.plot(w, t, linestyle="", marker="o", label="A={}".format(temp[1][:2]))

    plt.title("Tunneling probability Vs Barrier Width")
    plt.xlabel("Barrier width")
    plt.ylabel("Tunneling probability")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.savefig("Varying_Amplitude")
    plt.show()
    return 0


alltxt = glob.glob("*.txt")
gauss = []
for i in alltxt:
    if i.startswith(str("Gauss_")):
        gauss.append(i)

# load_gauss(gauss, log=True)

square_barrier()
