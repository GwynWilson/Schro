from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (d ** 2)) * np.exp(1j * k0 * x)


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


N = 2 ** 11
dx = 0.04
x = np.array([i * dx for i in range(N)])

hbar = 1
m = 1

bar_amp = 100
L = int(N / 1000)
x1 = N / 2 * dx
x2 = x1 + L * dx

ks = np.fft.fftfreq(N, dx / 2 * np.pi)
dk = -ks[0] + ks[1]

k0 = 10
x0 = int(N / 4) * dx
sig = 2

Psi_x = gauss_init(x, k0, x0=x0, d=sig)

V_x = barrier(x, bar_amp, x1, x2)
# V_x = np.zeros(N)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)

dt = 0.001
step = 50

c = Constants(bar_amp, dt, dx, k0)
# plt.plot(x, sch.psi_squared)
# plt.plot(x,V_x)
# plt.show()

a = Animate(sch, V_x, step, dt)
a.make_fig()
