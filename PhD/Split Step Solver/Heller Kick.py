import numpy as np
import matplotlib.pyplot as plt
from PhD.Stochastic.Heller import Heller as hl


def derivs(t, current, args, eta, dt):
    m, hbar = args
    x = current[1]
    v = 0
    a = (-2 * current[2] ** 2 / m)
    g = 1j * hbar * current[2] / m
    return x, v, a, g


N = 2 ** 11
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx
x0 = int(0.25 * x_length)

m = 1
hbar = 1
args = (m, hbar)

d = 1
k_initial = 4

p0 = k_initial * hbar
a0 = 1j * hbar / (4 * d ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)
init = [x0, p0 / m, a0, g0]

t = 0
dt = 0.01
step = 10
Ns = 100

Ntot = Ns * step
Nhalf = int(Ntot / 2)

init = [x0, p0, a0, g0]

Solver = hl.Heller(Nhalf, dt, init, derivs)
tl, xl, vl, al, gl = Solver.rk4(args)

init2 = [xl[-1], vl[-1] - 2 * p0, al[-1], gl[-1]]
Solver2 = hl.Heller(Nhalf, dt, init2, derivs)
tl2, xl2, vl2, al2, gl2 = Solver2.rk4(args)

tlcomb = np.concatenate((tl, np.asarray(tl2[1:]) + tl[-1]), axis=None)
xlcomb = np.concatenate((xl, xl2[1:]), axis=None)
vlcomb = np.concatenate((vl, vl2[1:]), axis=None)
alcomb = np.concatenate((al, al2[1:]), axis=None)
glcomb = np.concatenate((gl, gl2[1:]), axis=None)

Solver.tl = tlcomb
Solver.xl = xlcomb
Solver.vl = vlcomb
Solver.al = alcomb
Solver.gl = glcomb

Solver.plotBasic()
