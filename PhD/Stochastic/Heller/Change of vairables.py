from Heller import Heller
import matplotlib.pyplot as plt
import numpy as np


def derivs(t, current, args, N, dt):
    w, sig = args
    x = current[1] * dt
    v = -w ** 2 * current[0] * dt + w ** 2 * sig * N * np.sqrt(dt)
    a = 0
    g = 0
    return x, v, a, g


n = 1000
dt = 0.001
t = n * dt
init = [40, 0, 0, 0]
sig = 10
w = 10
args = (w, sig)

# solverStoc1 = Heller(n, dt, init, derivs)
# solverStoc1.averageRuns(1, args)
# print(solverStoc1.noise)
#
# plt.plot(solverStoc1.tl, solverStoc1.x_av-sig*solverStoc1.noise)
# plt.plot(solverStoc1.tl, solverStoc1.x_av)
# plt.show()


init = [0, 0, 0, 0]


def derivs(t, current, args, N, dt):
    w, sig = args
    x = (sig * N) ** 2 * dt
    v = sig * N * dt
    a = sig * N * np.sqrt(dt)
    g = 0
    return x, v, a, g


runs = 100
solverStoc1 = Heller(n, dt, init, derivs)
wsv = np.zeros(n, dtype=complex)
wsa = np.zeros(n, dtype=complex)
for i in range(runs):
    solverStoc1.rk4(args)
    wsv += (solverStoc1.vl) ** 2 / runs
    wsa += (solverStoc1.al) ** 2 / runs

plt.plot(solverStoc1.tl, wsv)
plt.plot(solverStoc1.tl, wsa)
plt.show()
