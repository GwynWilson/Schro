import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def wave_init(x, sig, x0):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - x0) ** 2) / (4 * sig ** 2))


def harmonic_potential_t(x, t, args=(1, 1, 1, 1)):
    return 0.5 * args[0] * args[1] ** 2 * (x - args[2] - args[3] * t**2) ** 2


def potential(x, t=0):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        temp[n] = 0.01 * t
    return temp


# Defining x axis
N = 2 ** 11
dx = 0.08
x_length = N * dx
x = np.linspace(0, x_length, N)
x0 = int(0.1 * x_length)

hbar = 1
m = 1
omega = 1
sigma = np.sqrt(hbar / (2 * m * omega))
a = 1
args = (m, omega, x0, a)

# Defining Psi and V
k_initial = 10
psi_x = wave_init(x, sigma, x0 - 10)
V_x = harmonic_potential_t(x, 0, args)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 5

sch = Schrodinger(x, psi_x, harmonic_potential_t, k, hbar=hbar, m=m, t=t, args=args)

"""
plt.plot(x, sch.mod_square_x(True))
plt.plot(x, V_x)
plt.ylim(0, max(np.real(psi_x)))
plt.show()
"""

a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_x)))),
            lim2=((ks[0], ks[N - 1]), (0, 30)))
a.make_fig()
