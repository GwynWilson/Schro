from Ito_Process import *
import matplotlib.pyplot as plt
import numpy as np


def a(x, t, args):
    omeg, gam = args
    return ((1j * omeg) - gam) * x


def aS(x, t, args):
    omeg, gam = args
    return ((1j * omeg) - gam) * x


def b(x, t, gam):
    return 1j * np.sqrt(2 * gam) * x


def z(t, args):
    omeg, gam, x0 = args
    return np.exp(((1j * omeg) - gam) * t) * x0


def complexOsc():
    processes = manyRuns(run, n, dt, a, b, x0=x0, aargs=(omeg, gam), bargs=gam)
    plotRuns(processes)

    zs = z(np.asarray(t_list), (omeg, gam, x0))
    plt.plot(np.real(itoAverages(processes)))
    plt.plot(zs)
    plt.show()


n = 1000
dt = 0.001
t_list = [i * dt for i in range(n)]

omeg = 10
gam = 1
x0 = 1

run = 5

processes1 = manyRuns(run, n, dt, a, b, x0=x0, aargs=(omeg, gam), bargs=gam)
processes2 = manyRuns(run, n * 10, dt / 10, a, b, x0=x0, aargs=(omeg, gam), bargs=gam)

zs = z(np.asarray(t_list), (omeg, gam, x0))
plt.plot([i * dt for i in range(n)], itoAverages(processes1), label="Run1")
plt.plot([i * dt / 10 for i in range(n * 10)], itoAverages(processes2), label="Run2")
plt.plot(t_list, zs, label="Exact")
plt.legend()
plt.savefig("Noisy Oscillator")
plt.show()
