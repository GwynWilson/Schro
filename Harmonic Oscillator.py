import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def wave_init(x, sig, x0):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - x0) ** 2) / (4 * sig ** 2))


def harmonic_potential(x, m, omeg, x0):
    return 0.5 * m * omeg ** 2 * (x - x0) ** 2


# Defining x axis
N = 2 ** 11
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x0 = int(x_length / 2)

# Constants
hbar = 1
m = 1
omega = 0.1
sigma = np.sqrt(hbar / (2 * m * omega))

# Defining Wavefunction
psi = wave_init(x, sigma, x0-2)
V_x = harmonic_potential(x, m, omega, x0)
#V_x = np.zeros(N)

# Defining k
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 10

sch = Schrodinger(x, psi, V_x, k, hbar, m, t)

a = Animate(sch, V_x, step, dt, lim1=((x[0], x[N - 1]), (0, max(psi))), lim2=((ks[0], ks[N - 1]), (0, 50)))
a.make_fig()
