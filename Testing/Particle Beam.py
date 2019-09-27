import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate

import time


def t_theory(L, V0, E, m=1, hbar=1):
    amp = 16 * E / V0 * (1 - (E / V0))
    coeff = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return amp * np.exp(-2 * L * coeff)


def t_theory2(L, V0, E, m=1, hbar=1):
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return (1 + 1 / 4 * (k1 / k2 + k2 / k1) ** 2 * np.sinh(k2 * L) ** 2) ** (-1)


def gauss_init(x, k0, x0=0, d=1):
    return np.exp(1j * k0 * x) + np.real(np.exp(-1j * k0 * x))
    # return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (d ** 2)) * np.exp(1j * k0 * x)


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


def theory(v0, E, L):
    const = 2 * m / (hbar ** 2)
    return np.exp(-2 * L * np.sqrt(const * (v0 - E)))


def run(x2, time=False):
    V_x = barrier(x, A, x1, x2)
    psi_init = gauss_init(x, k_init, x0=x0, d=sig)

    sch = Schrodinger(x, psi_init, V_x, k, hbar=hbar, m=m, t=0, args=x2)

    dat = []
    t_list = []
    for i in range(0, Ns):
        sch.evolve_t(step, dt)
        dat.append(sch.barrier_transmition())
        t_list.append(sch.t)

    # plt.plot(sch.k, abs(sch.psi_k))
    # plt.show()

    if time:
        return dat, t_list
    else:
        return dat


# Defining x axis
N = 2 ** 12
dx = 0.02
x_length = N * dx

x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx

hbar = 1
m = 1

# Barrier Definitions
A = 60
x1 = int(0.5 * N) * dx
L = 10 * dx
x2 = x1 + L
V_x = barrier(x, A, x1, x2)

# p0 = np.sqrt(2*m*A*0.2)
# dp2 = p0 * p0 * 1. / 80
# d = hbar / np.sqrt(2 * dp2)


# Wave Function definitions
x0 = int(0.35 * x_length)
sig = 4
k_init = 2
psi_init = gauss_init(x, k_init, x0=x0, d=sig)

#
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

t = 0
dt = 0.001
step = 50
Ns = 350

print(dt * Ns * step)

print("Diffusion", dt / (dx ** 2))
print("vdt", A * dt)
print("kdx", k_init * dx)
sch = Schrodinger(x, psi_init, V_x, k, hbar=hbar, m=m, t=t)

E = (hbar ** 2) * (k_init ** 2) / (2 * m)
# print("Energy", E)
# print("Energy_sim", sch.energy())
# print("Lenght", L)
# print("Theory", t_theory(L, A, E))

################### Testing
t_list = []
for i in range(0, Ns):
    t += dt * step
    t_list.append(t)
t = 0

L_list = range(1, 20, 2)
Tunnel_Val = []
T1_val = []
T2_val = []

L_list1 = np.linspace(1, 10, 100)

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.set_title("Tunneled Packet vs Time")
ax.set_xlabel("Time")
ax.set_ylabel("Probability of Particle Tunnelling")
for a in L_list:
    print(a)
    temp, t_list = run(x1 + a * dx, time=True)
    Tunnel_Val.append(temp[-1])
    ax.plot(t_list, temp, label='a={v}'.format(v=round(a, 3)))

save = [0 for k in range(len(L_list))]
for i in range(len(L_list)):
    save[i] = (L_list[i], Tunnel_Val[i])

np.savetxt("Square_Barrier_beam.txt", save)
var = [A, k_init, hbar, m, dx]
np.savetxt("Square_Barrier_var_beam.txt", var)

ax.legend()
plt.savefig('Tunneling Stuff')
plt.show()

theory = [t_theory(i * dx, A, E) for i in L_list1]
theory2 = [t_theory2(i * dx, A, E) for i in L_list1]

plt.plot(L_list, Tunnel_Val, label="simulation")
plt.plot(L_list1, theory, label="Emponential Theory")
plt.plot(L_list1, theory2, label="Sinh Theory")
plt.legend()
plt.savefig('Tunneling Stuff 2')
plt.show()

# ################# Initial
# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.show()

################ Animation
# a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_init)))),
#             lim2=((-30, 30), (0, 1)))
#
# a.make_fig()

################### Efficiency
# start = time.time()
# for i in range(Ns):
#     sch.evolve_t(step, dt)
# print(time.time()-start)
