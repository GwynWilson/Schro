from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


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


def t_theory2(L, V0, E, m=1, hbar=1):
    """Exact Tunneling"""
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    if V0 < E:
        k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    if abs(E - V0) < 10 ** (-15):
        diff = 0.00001
        k2 = np.sqrt(((2 * m) / (hbar ** 2)) * diff)
    else:
        k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (E - V0))
    return (1 + 1 / 4 * (k1 / k2 + k2 / k1) ** 2 * np.sinh(k2 * L) ** 2) ** (-1)


def t_choen2(L, V0, E, m=1, hbar=1):
    if E > V0:
        diff = (E - V0)
        arg = np.sqrt(2 * m * diff + 0j) * L / hbar
        trans = 4 * E * diff / (4 * E * diff + V0 ** 2 * np.sin(arg) ** 2)
    if abs(E - V0) < 10 ** (-15):
        diff = 0.00001
        arg = np.sqrt(2 * m * diff + 0j) * L / hbar
        trans = 4 * E * diff / (4 * E * diff + V0 ** 2 * np.sin(arg) ** 2)
    else:
        diff = (V0 - E)
        arg = np.sqrt(2 * m * diff + 0j) * L / hbar
        trans = 4 * E * diff / (4 * E * diff + V0 ** 2 * np.sinh(arg) ** 2)
    return np.real(trans)


def impedence(v, E, m=1, hbar=1):
    for n, i in enumerate(reversed(v)):
        diff = (E - i)
        if diff == 0:
            diff += 10 ** -99
        K = 1j * np.sqrt(2 * m * diff + 0j) / hbar
        z0 = -1j * hbar * K / m
        if n == 0:
            zload = z0
        else:
            zload = zin
        zin = z0 * ((zload * np.cosh(K * dx) - z0 * np.sinh(K * dx)) / (z0 * np.cosh(K * dx) - zload * np.sinh(K * dx)))

    coeff = np.real(((zin - z0) / (zin + z0)) * np.conj((zin - z0) / (zin + z0)))
    return 1 - coeff


def run(A):
    V_x = barrier(x, A, x1, x2)
    sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x2, t=0)
    while sch.t < finalt:
        sch.evolve_t(step, dt)
    T = sch.barrier_transmition()
    I = sch.impedencePacket()
    return T, I


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])

hbar = 1
m = 1

k0 = 2
x0 = int(N / 4) * dx
sig = 8

bar_amp = 5
L = 20
x1 = (N / 2) * dx
x2 = x1 + (L * dx)

Psi_x = gauss_init(x, k0, x0=x0, d=sig)

V_x = barrier(x, bar_amp, x1, x2)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x1)
print(sch.impedencePacket())

dt = 0.001
step = 500
finalt = 50

c = Constants(bar_amp, dt, dx, k0)

# plt.plot(x, sch.psi_squared)
# plt.plot(x, V_x)
# plt.show()

# a = Animate(sch, V_x, step, dt)
# a.make_fig()

######Testing Impedence
# E = np.arange(0, 20, 0.1)
# V0 = 5
#
# Vx1 = barrier(x, V0, x1, x2)
#
# theory = [t_choen2(x2 - x1, V0, i) for i in E]
# plt.plot(E, theory, label="Theory")
# imp1 = [impedence(Vx1, i) for i in E]
# plt.plot(E, imp1, label="Impedence 1")
# plt.legend()
# plt.savefig("Impedence_Square")
# plt.show()

#######Changing V
v_list = np.arange(1, 10, 1)
T_list = []
T2_list = []
T3_list = []
E = sch.energy()
for i in v_list:
    print("Run", i)
    T,I = run(i)
    T_list.append(T)
    T2_list.append(t_choen2(L * dx, i, E, m=m, hbar=hbar))
    T3_list.append(i)

plt.plot(v_list, T_list, label="Sim")
plt.plot(v_list, T2_list, label="Theory", linestyle="--")
plt.legend()
plt.xlabel("v0")
plt.ylabel("Transpmisson Probability")
plt.savefig("V0_Transmission_fine")
plt.show()
