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


m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34

Len = 2 * 10 ** - 6
lim = 4 * 10 ** - 5
N = 2**12
dx = 2 * lim / N

dt = 10 ** -7
dk = 2 * np.pi / (N * dx)

k_lim = np.pi / dx
k1 = -k_lim + (dk) * np.arange(N)

# x = np.array([i * dx for i in range(N)])
x = np.arange(-lim, lim, dx)

x0 = -1*10**-5
x1 = x[int(N / 2)]
x2 = x1 + Len

sig = 1 * 10 ** -6
k0 = 5 * 10 ** 6

E = (hbar ** 2 / 2 * m) * (k0 ** 2 + 1 / (4 * sig ** 2))
print(E)

Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = barrier(x, 10**-30, x1, x2)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, args=x1)

# plt.plot(x, sch.psi_squared)
plt.plot(x, V_x)
plt.show()

a = Animate(sch, V_x, 1000, dt, lim1=((-lim, lim), (0, max(sch.psi_squared))))
a.make_fig()
