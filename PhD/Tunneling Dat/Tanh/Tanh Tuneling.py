from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Input_Parameters_Realistic import *


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


def gaussian(x, mu, sig, A):
    raw = 1 / np.sqrt(2 * np.pi * (sig ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    return A * raw / max(raw)


def gaussianNonNorm(x, mu, sig, A):
    raw = 1 / np.sqrt(2 * np.pi * (sig ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    return A * raw


def squareBarrier(x, A, x1, x2):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        elif v > x2:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


def tanhBarrier(x, A, L, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    return A * raw / max(raw)


def tanhBarrierNoNorm(x, A, L, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    return A * raw


def tunnelingSpectra(V, E_list):
    sch = Schrodinger(x, psi_x, V, hbar=hbar, m=m, t=0, args=L / 2)
    T_list = []
    for e in E_list:
        T_list.append(sch.impedence(e))
    return T_list


v_sq = squareBarrier(x, bar_amp, -L / 2, L / 2)
v_tanh = tanhBarrier(x, bar_amp, L, 10 ** -8)

psi_x = gauss_init(x, k0, x0, d=sig)

sch = Schrodinger(x, psi_x, v_sq, hbar=hbar, m=m, t=0, args=L / 2)

ys = tanhBarrierNoNorm(x, bar_amp, L, 10 ** -6)

ppot, pcov = curve_fit(gaussian, x, ys, p0=(0, 2*10 ** -6, bar_amp))
print(ppot)
plt.plot(x / 10 ** -6, ys / scale * 10 ** -3, label="Tanh")
plt.plot(x / 10 ** -6, gaussian(x, ppot[0], ppot[1], ppot[2]) / scale * 10 ** -3, label="Fit")
plt.legend()
plt.xlim(-10, 10)
plt.ylabel("V (kHz)")
plt.xlabel("x (micrometers)")
plt.title("Tanh and Gaussian Comparison")
# plt.savefig("Gaussian Comparison No Norm")
plt.show()

# w_list = np.logspace(-6, -8, 5)
# w_list = np.asarray([10 ** -8, 10 ** -7, 0.3 * 10 ** -6, 10 ** -6])
#
# E_list = np.linspace(1, 4, 10000) * bar_amp
#
# for w in w_list:
#     plt.plot(x / 10 ** -6, tanhBarrier(x, bar_amp, L, w) / scale * 10 ** -3, label=f"{w}")
# plt.legend()
# plt.ylabel("V (kHz)")
# plt.xlabel("x (micrometers)")
# plt.title("Barrier Shape")
# plt.savefig("Tanh_w")
# plt.show()
#
# for w in w_list:
#     v_w = tanhBarrier(x, bar_amp, L, w)
#     plt.plot(E_list / bar_amp, tunnelingSpectra(v_w, E_list), label=f"{w}")
#
# plt.legend()
# plt.ylabel("Transmission Probability")
# plt.xlabel("E/V0")
# plt.title("Transmission Probability for changing w")
# plt.savefig("Tanh_w_tunnel")
# plt.show()
