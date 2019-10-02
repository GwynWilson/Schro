from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / 4 * (d ** 2)) * np.exp(1j * k0 * x)


def energy(k, hb, m):
    return (hb ** 2) * (k ** 2) / (2 * m)


def hbarChange(start, stop, step, show=False):
    hbar_range = np.arange(start, stop, step)

    E_list = energy(k0, hbar_range, m)
    Es_list = []
    for i in hbar_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=i, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(hbar_range))]

    plt.figure()
    plt.plot(hbar_range, E_list, label="Theoretical")
    plt.plot(hbar_range, Es_list, label="Simulated")
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
    plt.figure()
    plt.ylim(0.05, 0.07)
    plt.plot(hbar_range, Rel_list, label="Average.{}".format(round(np.mean(Rel_list), 5)))
    plt.title("Hbar Relative Difference")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("Hbar_rel.png")
    if show:
        plt.show()


def mChange(start, stop, step, show=False):
    m_range = np.arange(start, stop, step)

    E_list = energy(k0, hbar, m_range)
    Es_list = []
    for i in m_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=i, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(m_range))]

    plt.figure()
    plt.plot(m_range, E_list, label="Theoretical")
    plt.plot(m_range, Es_list, label="Simulated")
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
    plt.figure()
    plt.ylim(0.05, 0.07)
    plt.plot(m_range, Rel_list, label="Average.{}".format(np.mean(Rel_list)))
    plt.title("M Relative Difference")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("Relative_Energy_Difference")
    plt.savefig("M_rel.png")
    if show:
        plt.show()


def sigChange(start, stop, step, show=False):
    sig_range = np.arange(start, stop, step)

    E_ = energy(k0, hbar, m)
    Es_list = []
    for i in sig_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=i)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_ for j in range(len(sig_range))]

    plt.figure()
    plt.plot(sig_range, [E_ for i in sig_range], label="Theoretical")
    plt.plot(sig_range, Es_list, label="Simulated")
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

    Rel_list = [Diff_list[j] / E_ for j in range(len(sig_range))]
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

    E_list = energy(k_range, hbar, m)
    Es_list = []
    for i in k_range:
        Psi_x = gauss_init(x, i, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(k_range))]

    plt.figure()
    plt.plot(k_range, E_list, label="Theoretical")
    plt.plot(k_range, Es_list, label="Simulated")
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


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])
x = x - max(x) / 2

hbar = 1
m = 1

k0 = 2
x0 = x[int(N / 2)]
sig = 1

Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = np.zeros(N)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)

# plt.plot(x,sch.mod_square_x(r=True))
# plt.show()

E = energy(k0, hbar, m)
print("theoretical energy", E)
print("simulated energy", sch.energy())
print("diff",sch.energy()-E)

hbarChange(1, 15, 0.2)

mChange(1, 15, 0.2)

sigChange(1, 10, 0.2)

kChange(-5, 5, 0.1)

default_param = {"N": N, "dx": dx, "hbar": hbar, "m": m, "k0": k0, "m": m}

out = "param.txt"
fout = open(out, "w")
for k, v in default_param.items():
    fout.write(str(k) + " : " + str(v) + "\n")
fout.close()
