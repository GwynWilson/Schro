import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * x)


def x_pos(t, x0, kini, hbar=1, m=1):
    return x0 + t * kini * (hbar / m)


# Defining x axis
N = 2 ** 10
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x0 = int(0.4 * x_length)

# Defining Psi and V
k_initial = 22
psi_x = gauss_init(x, k_initial, x0, d=1)
V_x = np.zeros(N)

for n, v in enumerate(V_x):
    Nw = 0.01 * N
    if n < int(Nw) or n > int(N - Nw):
        V_x[n] = 1E100

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 2

sch = Schrodinger(x, psi_x, V_x, k)

plt.plot(x, sch.mod_square_x(True))
plt.plot(x, V_x)
plt.ylim(0, max(psi_x))
plt.show()

a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(psi_x))), lim2=((ks[0], ks[N-1]), (0, 30)),title='Finite Well')
a.make_fig()