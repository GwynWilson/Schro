import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def two_gauss(x, x1, x2, k1, k2, d=1):
    gauss1 = 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x1) ** 2 / (4 * d ** 2)) * np.exp(1j * k1 * x)
    gauss2 = 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x2) ** 2 / (4 * d ** 2)) * np.exp(1j * k2 * x)
    return gauss1 + gauss2


# Defining x axis
N = 2 ** 11
dx = 0.02
x_length = N * dx
x = np.linspace(0, x_length, N)
x1 = x[int(N / 4)]
x2 = x[int(3 * N / 4)]

# Defining Psi and V
k_initial = 4

psi_x = two_gauss(x, x1, x2, k_initial, -k_initial, d=1)
V_x = np.zeros(N)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 2

sch = Schrodinger(x, psi_x, V_x, k)

# plt.plot(sch.x, sch.mod_square_x(True))
# plt.show()

a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_x)+0.2))),
            lim2=((-2*k_initial, 2*k_initial), (0, abs(max(sch.psi_k)))), title='Two Packet Interferance')
a.make_fig()
