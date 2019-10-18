import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * x)


def barrier(x, A, w, x0):
    temp = A * (np.exp(-(x - x0) ** 2 / w ** 2))
    return temp


def noisyBarrier(x, t, args):
    A, w, x0, sig, mu = args
    r = sig * np.random.randn() + mu
    r = 1
    temp = A * r * (np.exp(-(x - x0) ** 2 / w ** 2))
    return temp


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])

hbar = 1
m = 1

# Barrier Definitions
A = 1
x1 = int(0.5 * N) * dx
W = 1
sig = 0.1
mu = 1
args = (A, W, x1, sig, mu)

k0 = 4
x0 = x[int(N / 4)]
sig = 4
k_init = 5

Psi_x = gauss_init(x, k0, x0=x0, d=sig)

sch = Schrodinger(x, Psi_x, noisyBarrier, hbar=hbar, m=m, args=args)
print("Width :", sig)
print(sch.x_width())
# print(sch.impedence())

dt = 0.001
step = 100

finalt = 40

c = Constants(A, dt, dx, k0)

a = Animate(sch, noisyBarrier, step, dt)
a.make_fig()
