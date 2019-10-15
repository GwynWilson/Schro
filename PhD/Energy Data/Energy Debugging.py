from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


def energy(k, hb, m):
    return (hb ** 2) * (k ** 2) / (2 * m)


def energy2(k, hb, m, sig):
    return (hb ** 2) * (k ** 2 + 1 / (4 * sig ** 2)) / (2 * m)


def hbarChange(start, stop, step, show=False):
    hbar_range = np.arange(start, stop, step)

    E_list = energy2(k0, hbar_range, m, sig)
    Es_list = []
    for i in hbar_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=i, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(hbar_range))]

    plt.figure()
    plt.plot(hbar_range, E_list, label="Theoretical")
    plt.plot(hbar_range, Es_list, label="Simulated", linestyle="--")
    plt.title("Hbar")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Energy")
    plt.savefig("Hbar.png")
    if show:
        plt.show()

    plt.figure()
    plt.plot(hbar_range, Diff_list, label="Diff")
    plt.title("Hbar_Diff")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Energy_Difference")
    plt.savefig("Hbar_diff.png")
    if show:
        plt.show()

    Rel_list = [Diff_list[j] / E_list[j] for j in range(len(hbar_range))]
    av = np.mean(Rel_list)
    plt.figure()
    plt.ylim(av - 0.1, av + 0.1)
    plt.plot(hbar_range, Rel_list, label="Average.{}".format(round(av, 5)))
    plt.title("Hbar Relative Difference")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("Hbar_rel.png")
    if show:
        plt.show()


def mChange(start, stop, step, show=False):
    m_range = np.arange(start, stop, step)

    E_list = energy2(k0, hbar, m_range, sig)
    Es_list = []
    for i in m_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=i, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(m_range))]

    plt.figure()
    plt.plot(m_range, E_list, label="Theoretical")
    plt.plot(m_range, Es_list, label="Simulated", linestyle="--")
    plt.title("M")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("Energy")
    plt.savefig("M.png")
    if show:
        plt.show()

    plt.figure()
    plt.plot(m_range, Diff_list, label="Diff")
    plt.title("M_Diff")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("Energy_Difference")
    plt.savefig("M_diff.png")
    if show:
        plt.show()

    Rel_list = [Diff_list[j] / E_list[j] for j in range(len(m_range))]
    av = np.mean(Rel_list)
    plt.figure()
    plt.ylim(av - 0.1, av + 0.1)
    plt.plot(m_range, Rel_list, label="Average.{}".format(round(av, 5)))
    plt.title("M Relative Difference")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("M_rel.png")
    if show:
        plt.show()


def sigChange(start, stop, step, show=False):
    sig_range = np.arange(start, stop, step)

    E_ = energy2(k0, hbar, m, sig_range)
    Es_list = []
    for i in sig_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=i)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_[j] for j in range(len(sig_range))]

    plt.figure()
    plt.plot(sig_range, E_, label="Theoretical")
    plt.plot(sig_range, Es_list, label="Simulated", linestyle="--")
    plt.title("Sigma")
    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("Energy")
    plt.savefig("Sigma.png")
    if show:
        plt.show()

    plt.figure()
    plt.plot(sig_range, Diff_list, label="Diff")
    plt.title("Sigma_Diff")
    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("Energy_Difference")
    plt.savefig("Sigma_diff.png")
    if show:
        plt.show()

    Rel_list = [Diff_list[j] / E_[j] for j in range(len(sig_range))]
    plt.figure()
    plt.plot(sig_range, Rel_list, label="Relative Difference")
    plt.title("Sigma Relative Difference")
    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("Sig_rel.png")
    if show:
        plt.show()


def kChange(start, stop, step, show=False):
    k_range = np.arange(start, stop, step)

    E_list = energy2(k_range, hbar, m, sig)
    Es_list = []
    for i in k_range:
        Psi_x = gauss_init(x, i, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(k_range))]

    plt.figure()
    plt.plot(k_range, E_list, label="Theoretical")
    plt.plot(k_range, Es_list, label="Simulated", linestyle="--")
    plt.title("k0")
    plt.legend()
    plt.xlabel("k0")
    plt.ylabel("Energy")
    plt.savefig("k0.png")
    if show:
        plt.show()

    plt.figure()
    plt.plot(k_range, Diff_list, label="Diff")
    plt.title("k0_Diff")
    plt.legend()
    plt.xlabel("k0")
    plt.ylabel("k0_Difference")
    plt.savefig("k0_diff.png")
    if show:
        plt.show()

    # Rel_list = [Diff_list[j] / E_list[j] for j in range(len(k_range))]
    # plt.figure()
    # plt.plot(k_range, Rel_list, label="Relative Difference")
    # plt.title("k0 Relative Difference")
    # plt.legend()
    # plt.xlabel("k0")
    # plt.ylabel("Relative_Energy_Difference")
    # plt.savefig("k0_rel.png")
    # if show:
    #     plt.show()


def Nchange(nrange, show=False):
    E_ = energy2(k0, hbar, m, sig)
    Es_list = []
    for i in nrange:
        N = 2 ** i
        x = np.array([i * dx for i in range(N)])
        x = x - max(x) / 2

        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        V_x = np.zeros(N)

        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_ for j in range(len(nrange))]

    plt.figure()
    plt.plot(nrange, [E_ for i in nrange], label="Theoretical")
    plt.plot(nrange, Es_list, label="Simulated", linestyle="--")
    plt.title("N")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Energy")
    plt.savefig("N.png")
    if show:
        plt.show()

    plt.figure()
    plt.ylim()
    plt.plot(nrange, Diff_list, label="Diff")
    plt.title("N_Diff")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Energy_Difference")
    plt.savefig("N_diff.png")
    if show:
        plt.show()

    Rel_list = [Diff_list[j] / E_ for j in range(len(nrange))]
    plt.figure()
    plt.plot(nrange, Rel_list, label="Relative Difference")
    plt.title("N Relative Difference")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("N_rel.png")


def xchange(xrange, show=False):
    E_ = energy2(k0, hbar, m, sig)
    Es_list = []
    for i in xrange:
        dx = i
        x = np.array([i * dx for i in range(N)])
        x = x - max(x) / 2

        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        V_x = np.zeros(N)

        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_ for j in range(len(xrange))]

    plt.figure()
    plt.plot(xrange, [E_ for i in xrange], label="Theoretical")
    plt.plot(xrange, Es_list, label="Simulated", linestyle="--")
    plt.title("dx")
    plt.legend()
    plt.xlabel("dx")
    plt.ylabel("Energy")
    plt.savefig("dx.png")
    if show:
        plt.show()

    plt.figure()
    plt.plot(xrange, Diff_list, label="Diff")
    plt.title("dx_Diff")
    plt.legend()
    plt.xlabel("dx")
    plt.ylabel("Energy_Difference")
    plt.savefig("dx_diff.png")
    if show:
        plt.show()

    Rel_list = [Diff_list[j] / E_ for j in range(len(xrange))]
    plt.figure()
    plt.plot(xrange, Rel_list, label="Relative Difference")
    plt.title("dx Relative Difference")
    plt.legend()
    plt.xlabel("dx")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("dx_rel.png")


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])
x = x - max(x) / 2

hbar = 1
m = 1

k0 = 1
x0 = x[int(N / 2)]
sig = 1

Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = np.zeros(N)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)

# plt.plot(x,sch.mod_square_x(r=True))
# plt.show()

E = energy(k0, hbar, m)
Es = sch.energy()
print("theoretical energy", E)
print("additional theoretical energy", energy2(k0, hbar, m, sig))
print("simulated energy", Es)
print("diff", Es - E)

hbarChange(1, 15, 0.2)

mChange(1, 15, 0.2)

sigChange(1, 10, 0.2)

kChange(-5, 5, 0.1)

Nchange([9, 10, 11, 12, 13])

xchange(np.linspace(0.01, 0.4, 10))


default_param = {"N": N, "dx": dx, "hbar": hbar, "m": m, "k0": k0, "m": m}

out = "param.txt"
fout = open(out, "w")
for k, v in default_param.items():
    fout.write(str(k) + " : " + str(v) + "\n")
fout.close()
