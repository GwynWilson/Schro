from PhD.Stochastic.Heller import Heller as hel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def derivsStocdt(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1] * dt
    v = -w ** 2 * (current[0] * dt - eta * sig)
    a = ((-2 * current[2] ** 2 / m) - (m * w ** 2) / 2) * dt
    g = (1j * hbar * current[2] / m + m * current[1] ** 2 / 2) * dt - 0.5 * m * w ** 2 * current[
        0] ** 2 * dt + m * w ** 2 * current[0] * eta * sig - 0.5 * m * w ** 2 * sig ** 2 * eta ** 2
    return x, v, a, g


def expected(t, args, init):
    t = np.asarray(t)
    x0, v0, a0, g0 = init
    w, m, hbar, sig = args
    # return g0 + 1j*hbar*a0*t/m +m*w**2*sig**2*np.sin(2*w*t)/4 - m*w**2*sig**2*t/2
    return g0 + 1j * hbar * a0 * t / m + m * w ** 2 * sig ** 2 * (1-
                np.cos(2 * w * t)) / 8 - m * w ** 2 * sig ** 2 * t / 2


def derivsTest(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1] * dt
    v = -w ** 2 * (current[0] * dt - eta * sig)
    a = ((-2 * current[2] ** 2 / m) - (m * w ** 2) / 2) * dt
    g = 0.5 * m * current[1] ** 2 * dt - 0.5 * m * w ** 2 * current[0] ** 2 * dt
    return x, v, a, g


def testExpect(t, args, init):
    t = np.asarray(t)
    x0, v0, a0, g0 = init
    w, m, hbar, sig = args
    # return g0 + 1j*hbar*a0*t/m +m*w**2*sig**2*np.sin(2*w*t)/4 - m*w**2*sig**2*t/2
    # return g0+0.5*m*w**4*sig**2*(t**2/4+(1-np.cos(2*w*t))/(8*w**2))
    return g0 + m * w ** 4 * sig ** 2 * (1 - np.cos(2 * w * t)) / (8 * w ** 2)


n = 10000
dt = 0.0001

w = 10
sig = 1
m = 1
hbar = 1
args = (w, m, hbar, sig)
temp = 0.5 * m * w ** 2

a0 = 1j * m * w / 2
init = [0, 0, a0, 0]

############################### Gamma Calculation
n_runs = 5000
title = "Gamma_full"

Heller = hel.Heller(n, dt, init, derivsStocdt)
Heller.averageRuns(n_runs,args,dtver=True,save=title)


plt.plot(Heller.tl,Heller.g_av,label="Numeric")
plt.plot(Heller.tl,expected(Heller.tl,args,init),label="Analytic")
plt.title(f"Stochastic Harmonic Oscillator Phase, n={n_runs}")
plt.xlabel("Time")
plt.ylabel("Gamma")
plt.legend()
plt.savefig(f"Gamma_n_{n_runs}")
plt.show()

# loaded = np.load(f"{title}_{n_runs}.npz")
# tl = loaded["tl"]
# g_av = loaded["g_av"]
#
# plt.plot(tl,g_av)
# plt.plot(tl, expected(tl,args,init))
# plt.show()


################# Individual Testing
# n_runs = 1000
# Heller = hel.Heller(n, dt, init, derivsTest)
# Heller.averageRuns(n_runs, args, dtver=True)
#
# # plt.plot(Heller.tl,Heller.v_av)
# # plt.show()
#
#
# plt.plot(Heller.tl, Heller.g_av, label="Numeric")
# plt.plot(Heller.tl, testExpect(Heller.tl, args, init), label="Analytic")
# plt.title(f"Stochastic Harmonic Oscillator Phase, n={n_runs}")
# plt.xlabel("Time")
# plt.ylabel("Gamma")
# plt.legend()
# # plt.savefig(f"Gamma_n_{n_runs}")
# plt.show()
