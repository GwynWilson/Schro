import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Numerical_Constants import Constants
from Animate import Animate


def x_pos(t, A, omega, x0):
    return -A * np.cos(omega * t) + x0


def wave_init(x, sig, xA):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - xA) ** 2) / (4 * (sig ** 2)))


def harmonic_potential(x, m=1, w=1):
    return 0.5 * m * (w ** 2) * (x ** 2)


# Defining x axis
N = 256
xmax = 4 * 10 ** -5
Ntper1 = 10 ** 4
tmax = 1
nsave = 10
Nt = tmax * Ntper1
dt = tmax / Nt
dx = 2 * xmax / N
dk = 2 * np.pi / (N * dx)

x = dx * np.arange(-N / 2, N / 2)

x0 = 2 * 10 ** -5
hbar = 1.0545718 * 10 ** -34
m = 1.44316072 * 10 ** -25
omegax = 80 * np.pi
sigma = 1 * 10 ** -6

t = 0

psi_init = wave_init(x, sigma, -x0)
V = harmonic_potential(x, m=m, w=omegax)
print(V[0])

plt.plot(x, psi_init)
plt.plot(x, V)
plt.show()

sch = Schrodinger(x, psi_init, V, hbar=hbar, m=m, t=0)


# a = Animate(sch, V, 1, dt,lim1=((-xmax, xmax), (0, max(sch.psi_squared))))
# a.make_fig()

simx = []
tl = []
El = []

temp = 0
while temp < Nt:
    sch.evolve_t(1, dt=dt)
    temp += 1
    if (temp % nsave) == 0:
        simx.append(sch.expectation_x())
        tl.append(temp * dt)
        El.append(sch.energy())

actualx = x_pos(np.asarray(tl), x0, omegax, 0)
diff = abs(np.asarray(simx) - actualx)

plt.plot(tl, simx)
plt.plot(tl, actualx, linestyle="--")
plt.show()

plt.plot(tl, diff)
plt.show()

