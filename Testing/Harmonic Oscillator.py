import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def x_pos(t, A, omega, x0):
    return -A * np.cos(omega * t) + x0


def wave_init(x, sig, xA):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - xA) ** 2) / (4 * sig ** 2))


def harmonic_potential(x, x0, m=1, w=1):
    return 0.5 * m * w ** 2 * (x - x0) ** 2


# Defining x axis
N = 2 ** 12
dx = 0.05
x_length = N * dx
x0 = int(0.5 * x_length)

x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx - x0

x0 = 0

hbar = 1
m = 1
omega = 1
sigma = np.sqrt(hbar / (2 * m * omega))

A = int(0.1 * x_length)
xA = x0 - A

dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

t = 0
dt = 0.01
step = 5
Ns = 300
print(Ns * step * dt)

psi_init = wave_init(x, sigma, xA)
V = harmonic_potential(x, x0, m=m, w=omega)

# plt.plot(x,psi_init)
# plt.plot(x, V)
# plt.show()

sch = Schrodinger(x, psi_init, V, k, hbar=hbar, m=m)

a = Animate(sch, V, step, dt, lim1=((-x_length/4, x_length/4), (0, max(np.real(psi_init)))),
            lim2=((ks[0], ks[N - 1]), (0, 1)), title="Harmonic Oscillator", frames=Ns)
a.make_fig()

# t_list = []
# expec_x = []
#
# for i in range(Ns):
#     if i != 0:
#         sch.evolve_t(step, dt)
#     t_list.append(sch.t)
#     expec_x.append(sch.expectation_x())
#
# x_pos_list = [x_pos(j, A, omega, x0) for j in t_list]
# xdiff = [np.abs(expec_x[n] - x_pos_list[n]) for n in range(len(expec_x))]
#
# plt.plot(t_list, expec_x, label='Simulated x')
# plt.plot(t_list, x_pos_list, linestyle='--', label='Theoretical x')
# plt.title('Expectation value of x over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('HO.png')
# plt.show()
#
# plt.plot(t_list, xdiff)
# plt.title('Difference')
# plt.xlabel('Time')
# plt.ylabel(r'$x - <x>$')
# plt.savefig('HO_diff.png')
# plt.show()
