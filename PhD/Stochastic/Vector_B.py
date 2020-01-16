import Ito_Process as ito
import matplotlib.pyplot as plt
import numpy as np


def a(x, t):
    return 0


def b(x, t):
    return amp


def oneRun(n, dt, a, b, x0, B1_mag, B2_mag, thet):
    run1 = ito.itoProcess(n, dt, a, b, x0=x0)
    B1_x = B1_mag * np.asarray(run1)
    B1_y = np.zeros(n)

    run2 = ito.itoProcess(n, dt, a, b, x0=x0)
    B2_x = B2_mag * np.asarray(run2) * np.cos(thet)
    B2_y = B2_mag * np.asarray(run2) * np.sin(thet)
    return B1_x + B2_x, B1_y + B2_y


I = 1
amp = 10 ** (-5)

time = 10
dt = 10 ** -3
n = int(time / dt)

Bmag = 10
thet = np.pi / 4

Av_x = Bmag * I + Bmag * I * np.cos(thet)
Av_y = Bmag * I * np.sin(thet)

nRuns = 10
for i in range(nRuns):
    runx, runy = oneRun(n, dt, a, b, I, Bmag, Bmag, thet)
    plt.scatter(runx, runy)
plt.scatter(Av_x,Av_y)
plt.show()
