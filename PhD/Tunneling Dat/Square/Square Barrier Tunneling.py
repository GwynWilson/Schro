from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt
import time


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
        diff = 10 ** -99
        arg = np.sqrt(2 * m * diff + 0j) * L / hbar
        trans = 4 * E * diff / (4 * E * diff + V0 ** 2 * np.sin(arg) ** 2)
    else:
        diff = (V0 - E)
        arg = np.sqrt(2 * m * diff + 0j) * L / hbar
        trans = 4 * E * diff / (4 * E * diff + V0 ** 2 * np.sinh(arg) ** 2)
    return np.real(trans)


def minT(E, V):
    return 4 * E * (E - V) / (4 * E * (E - V) + V ** 2)


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


def run(A, choen=False):
    V_x = barrier(x, A, x1, x2)
    sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x2, t=0)
    I = sch.impedencePacket()
    while sch.t < finalt:
        print("Height :{h} Time:{t}".format(h=A, t=sch.t))
        sch.evolve_t(step, dt)
    T = sch.barrier_transmition()
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
print(sch.energy())

dt = 0.001
step = 100
finalt = 50

c = Constants(bar_amp, dt, dx, k0)


# plt.plot(x, sch.psi_squared)
# plt.plot(x, V_x)
# plt.show()

# a = Animate(sch, V_x, step, dt)
# a.make_fig()

######Testing Impedence
# krange = np.arange(1, 2, 1)
# tol = np.logspace(-7, -11, 100)
# V0 = 5
# Vx1 = barrier(x, V0, x1, x2)
#
# Transmission = []
# timer = []
# for tolerance in tol:
#     print(tolerance)
#     psi = gauss_init(x, 4, x0=x0, d=sig)
#     sch = Schrodinger(x, psi, Vx1, hbar=hbar, m=m)
#     start = time.clock()
#     imp = sch.impedencePacket(tol=tolerance)
#     stop = time.clock()
#
#     timer.append(stop - start)
#     Transmission.append(imp)
#
# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.semilogx(tol, Transmission, label="Impedence 1", marker="x", linestyle="none")
# ax1.legend()
#
# ax2.semilogx(tol, timer)
# plt.show()

#######Changing V
# v_list = np.arange(0.5, 5, 0.1)
# print(v_list)
# T_list = []
# T2_list = []
# T3_list = []
# E = sch.energy()
# for i in v_list:
#     print("Run", i)
#     T, I = run(i, choen=True)
#     T_list.append(T)
#     T2_list.append(t_choen2(L * dx, i, E, m=m, hbar=hbar))
#     T3_list.append(I)
#
# save = [0 for k in range(len(v_list))]
# for i in range(len(v_list)):
#     save[i] = (v_list[i], T_list[i], T2_list[i], T3_list[i])
#
# var = [N, dx, L * dx, dt, k0]
#
# np.savetxt("Square_Barrier.txt", save)
# np.savetxt("Square_Barrier_var.txt", var)

# plt.plot(v_list, T_list, label="Sim")
# plt.plot(v_list, T2_list, label="Theory", linestyle="--")
# plt.plot(v_list, T3_list, label="Impedence", linestyle="--")
# plt.legend()
# plt.xlabel("v0")
# plt.ylabel("Transpmisson Probability")
# plt.savefig("V0_Transmission")
# plt.show()

############ Resonances
def n(L, V, E):
    k2 = np.sqrt(2 * m * (E - V)) / hbar
    output = []
    arg = 0
    c = 0
    while arg < L:
        c += 1
        arg = c * np.pi / k2 - np.pi / 2 * k2
        output.append(arg)
    return output


V = 1
E = np.linspace(1, 5, 100) * V
L = 10
V_x = barrier(x, V, x1, x2)

plt.plot(x,V_x)
plt.show()

Transmission = [t_choen2(L*dx, V, e) for e in E]
# for e in E:
#     ns = n(L, V, e)
Imp = [impedence(V_x, e) for e in E]



plt.plot(E, Transmission)
plt.plot(E, Imp)
plt.show()
