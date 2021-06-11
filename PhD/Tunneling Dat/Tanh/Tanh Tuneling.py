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


def Impedence(v, E, m=1, hbar=1):
    for n, i in enumerate(reversed(v)):
        diff = (E - i)
        if diff == 0:
            diff += 10 ** -99
        K = 1j * np.sqrt(2 * m * diff + 0j) / hbar
        z0 = -1j * hbar * K / m
        if n == 0:
            zload = z0
        else:
            zload = zin
        zin = z0 * ((zload * np.cosh(K * dx) - z0 * np.sinh(K * dx)) / (
                z0 * np.cosh(K * dx) - zload * np.sinh(K * dx)))

    coeff = np.real(((zin - z0) / (zin + z0)) * np.conj((zin - z0) / (zin + z0)))
    return 1 - coeff


e = bar_amp
l = np.sqrt(hbar ** 2 / (2 * m * bar_amp))
L = 10 * l

x = np.linspace(-20, 20, 1000) * l
dx = x[1] - x[0]

v_sq = squareBarrier(x, bar_amp, -L / 2, L / 2)
v_tanh = tanhBarrier(x, bar_amp, L, 10 ** -8)

psi_x = gauss_init(x, k0, x0, d=sig)

sch = Schrodinger(x, psi_x, v_sq, hbar=hbar, m=m, t=0, args=L / 2)

ys = tanhBarrierNoNorm(x, bar_amp, L, 10 ** -6)

# ppot, pcov = curve_fit(gaussian, x, ys, p0=(0, 2*10 ** -6, bar_amp))
# plt.plot(x / 10 ** -6, ys / scale * 10 ** -3, label="Tanh")
# plt.plot(x / 10 ** -6, gaussian(x, ppot[0], ppot[1], ppot[2]) / scale * 10 ** -3, label="Fit")
# plt.legend()
# plt.xlim(-10, 10)
# plt.ylabel("V (kHz)")
# plt.xlabel("x (micrometers)")
# plt.title("Tanh and Gaussian Comparison")
# plt.savefig("Gaussian Comparison No Norm")
# plt.show()

# w_list = np.logspace(-6, -8, 5)


# w_list = np.asarray([10 ** -8, 10 ** -7, 0.3 * 10 ** -6, 10 ** -6])
w_list = np.asarray([0.01 * L / 2, 0.1 * L / 2, 0.3 * L / 2, L / 2])

########### Barrier Shape
# E_list = np.linspace(1, 4, 10000) * bar_amp
#
# plt.rcParams.update({"font.size":14})
# for w in w_list:
#     plt.plot(x / l, tanhBarrier(x, bar_amp, L, w) / bar_amp, label=fr"${w/l:.2f}$")
# plt.legend()
# plt.xlim(-20,20)
# plt.ylabel(r"$V(x)/V_0$")
# plt.xlabel(r"$x/l$")
# plt.ylim(0,1.05)
# # plt.title("Barrier Shape")
# plt.tight_layout()
# plt.savefig("Tanh_w")
# plt.show()


############ Energy Tuneling
# E_list = np.linspace(0.75, 2, 500) * bar_amp
# # E_list = np.linspace(1.12, 1.2, 300) * bar_amp
#
# plt.rcParams.update({"font.size":14})
# for w in w_list:
#     v_w = tanhBarrier(x, bar_amp, L, w)
#     plt.plot(E_list / bar_amp, tunnelingSpectra(v_w, E_list), label=fr"${w/l:.2f}$")
#
# plt.legend()
# plt.ylabel("Transmission Probability")
# plt.xlabel(r"$E/V_0$")
# plt.xlim(min(E_list/bar_amp),max(E_list/bar_amp))
# # plt.xlim(0.75,2)
# plt.ylim(0,1.05)
# plt.tight_layout()
# # plt.savefig("Tanh_w_tunnel")
# plt.show()

#### a Tunneling
L_list = np.linspace(0.5, 1.5, 200) * L

L_list = np.linspace(0.9, 1, 200) * L
w_list = np.asarray([0.01 * L / 2, 0.1 * L / 2, L / 2])

plt.rcParams.update({"font.size": 14})
for w in w_list:
    temp = []
    for l_i in L_list:
        v_w = tanhBarrier(x, bar_amp, l_i, w)
        imp = Impedence(v_w, 2 * bar_amp, m=m, hbar=hbar)
        temp.append(imp)
    plt.plot(L_list / l, temp, label=fr"${w/l:.2f}$")

plt.legend()
plt.ylabel("Transmission Coefficient")
plt.xlabel(r"$L_t/l$")
plt.xlim(min(L_list/l),max(L_list/l))
plt.xlim(9, 10)
plt.ylim(0.993, 1.0005)
plt.tight_layout()
# plt.savefig("Tanh_a")
plt.savefig("Tanh_a_zoom")
plt.show()
