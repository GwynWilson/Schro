from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-((x - x0) ** 2) / (d ** 2)) * np.exp(1j * k0 * x)


def barrier(x, A, x1, x2):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        elif v > x2:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


def stepv(x, A, x1):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


def theoryT(V0, E, m=1, hbar=1):
    """
    Step UP
    """
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (E - V0))
    if E > V0:
        return (4 * k1 * k2) / ((k1 + k2) ** 2)
    else:
        return 0


def t_theory2(L, V0, E, m=1, hbar=1):
    """Exact Tunneling"""
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return (1 + 1 / 4 * (k1 / k2 + k2 / k1) ** 2 * np.sinh(k2 * L) ** 2) ** (-1)


def run(A):
    V_x = barrier(x, A, x1, x2)
    sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x2, t=0)
    while sch.t < finalt:
        sch.evolve_t(step, dt)

    # plt.plot(sch.x, sch.psi_squared)
    # plt.show()

    T = sch.barrier_transmition()
    return T


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])

hbar = 1
m = 1

k0 = 2
x0 = int(N / 4) * dx
sig = 8

bar_amp = 3
L = int(7/k0)
x1 = N / 2 * dx
x2 = x1 + L * dx

ks = np.fft.fftfreq(N, dx / 2 * np.pi)
dk = -ks[0] + ks[1]



Psi_x = gauss_init(x, k0, x0=x0, d=sig)

V_x = barrier(x, bar_amp, x1, x2)
# V_x = np.zeros(N)
# V_x = stepv(x, bar_amp, x1)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x1)

dt = 0.01
step = 50

finalt = 40

c = Constants(bar_amp, dt, dx, k0)

E = (hbar ** 2) * (k0 ** 2) / (2 * m)
print("Energy", E)
print("Energy_sim", sch.energy())
print("V0", bar_amp)

# plt.plot(x, sch.psi_squared)
# plt.plot(x,V_x)
# plt.show()

# a = Animate(sch, V_x, step, dt)
# a.make_fig()

v_list = np.arange(1, 3, 0.25)
T_list = []
T2_list = []
for i in v_list:
    print("Run", i)
    T_list.append(run(i))
    T2_list.append(t_theory2(L, i, E, m=m, hbar=hbar))

plt.plot(v_list, T_list, label="Sim")
plt.plot(v_list, T2_list, label="Theory")
plt.legend()
plt.show()
