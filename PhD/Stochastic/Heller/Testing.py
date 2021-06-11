import numpy as np
import matplotlib.pyplot as plt
from Heller import Heller
from scipy.integrate import simps


def euler(Nstep, dt, sig):
    x0 = 0
    list_ = []
    for i in range(Nstep):
        list_.append(x0)
        x0 += np.random.randn() * np.sqrt(dt)
    return list_


def derivsStoc(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1]
    v = -w ** 2 * (current[0] - eta * sig)
    a = (-2 * current[2] ** 2 / m) - (m * w ** 2) / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * (current[0] - eta * sig) ** 2 / 2
    return x, v, a, g


def expectSquareWhite(t, args, init):
    w, m, hbar, sig = args
    t = np.asarray(t)
    xs = w ** 2 * sig ** 2 * 0.5 * (t - np.sin(2 * w * t) / (2 * w))
    vs = w ** 4 * sig ** 2 * 0.5 * (t + np.sin(2 * w * t) / (2 * w))
    return xs, vs


def expectSquareBrown(t, args, init):
    w, m, hbar, sig = args
    t = np.asarray(t)
    xs = w ** 2 * sig ** 2 * (3 * t ** 2 / (4 * w ** 2) + np.sin(2 * w * t) / (8 * w ** 3) - np.sin(w * t) / (w ** 3))
    vs = w ** 4 * sig ** 2 * 0.5 * (t + np.sin(2 * w * t) / (2 * w))
    return xs, vs


# dt = 0.1
# N = 1000
# t = [i * dt for i in range(N)]
# euler_ = euler(N, dt, 1)
# plt.plot(t, euler_)
# plt.show()

n = 1000
dt = 0.001

w = 10
sig = 10
m = 1
hbar = 1
args = (w, m, hbar, sig)
temp = 0.5 * m * w ** 2

a0 = 1j * m * w / 2
init = [0, 0, a0, 0]

noise = np.cumsum(np.random.randn(n) * np.sqrt(dt))

solverStoc = Heller(n, dt, init, derivsStoc)
####### Average
# solverStoc.averageRuns(10000, args,noise=noise)
# solverStoc.plotBasic(average=True, title="Brownian Noise Oscillator")


####### Square
solverStoc.averageRuns(10000, args, noise=noise, square=True)
solverStoc.plotSquare(title="Brownian Noise Squared")


###### Single Run
# noise = np.random.randn(n) * np.sqrt(dt)
# tl, xl, vl, al, gl = solverStoc.rk4(args, noise=noise)
# print(tl)
#
# t_final = tl[-1]
# integrand = []
# for n, v, in enumerate(np.asarray(tl)):
#     integrand.append(np.sin(w * (t_final - v)) * noise[n])
#
# x_expect = w * sig * np.cumsum(integrand)
#
# plt.plot(tl, xl)
# # plt.plot(tl,x_expect)
# plt.show()
#
# plt.plot(tl, x_expect)
# plt.show()
