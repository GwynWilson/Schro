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
    print(hbar_range)

    E_list = energy(k0, hbar_range, m)
    Es_list = []
    for i in hbar_range:
        Psi_x = gauss_init(x, k0, x0=x0, d=sig)
        sch = Schrodinger(x, Psi_x, V_x, hbar=i, m=m, t=0)
        Es_list.append(sch.energy())

    Diff_list = [Es_list[j] - E_list[j] for j in range(len(hbar_range))]

    plt.plot(hbar_range, E_list, label="Theoretical")
    plt.plot(hbar_range, Es_list, label="Simulated")
    plt.title("Hbar")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Energy")
    plt.savefig("Hbar.png")
    if show:
        plt.show()

    plt.figure(0)
    plt.plot(hbar_range, Diff_list, label="Diff")
    plt.title("Hbar_Diff")
    plt.legend()
    plt.xlabel("Hbar")
    plt.ylabel("Energy_Difference")
    plt.savefig("Hbar_diff.png")
    if show:
        plt.show()


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

E = (hbar ** 2) * (k0 ** 2) / (2 * m)

# print("Theoretical Energy", E)
# print("Simulated E", sch.energy())
# print("Difference", sch.energy() - E)

# plt.plot(x,sch.mod_square_x(r=True))
# plt.show()

hbarChange(1,10,0.5)

