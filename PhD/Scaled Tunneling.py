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



m = 1.44 * 10 ** -24
hbar = 1.054 * 10 ** -34
v0 = 30*10**3 * 2 * np.pi * hbar

v0 = 30*10**3
E = 5000 * 2 * np.pi * hbar
# dx = 4 * 10 ** - 6
Len = 2 * 10 ** - 6
lim = 15 * 10 ** -6
N = 2 ** 12
dx = 2*lim/N


# x = np.array([i * dx for i in range(N)])
x = np.arange(0, 2*lim, dx)
N = len(x)
print(len(x))

x0 = x[int(N / 4)]
x1 = x[int(N / 2)]
x2 = x1 + Len

sig = 1 * 10 ** -6
k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))

Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = barrier(x, v0, x1, x2)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x1)

# plt.plot(x, sch.psi_squared)
# plt.plot(x, V_x)
# plt.show()


dt = 10**-6
step = 100

c = Constants(v0, dt, dx, k0)


a = Animate(sch, V_x, step, dt)
a.make_fig()