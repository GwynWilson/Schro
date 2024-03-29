import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def barrier(x, A, x1, x2):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        elif v > x2:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


def gaussian(x, V0, L, sig=1, mu=0):
    raw = 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    norm = simps(raw, x)
    area = V0 * L
    raw = raw * area / norm
    return raw


def gaussianNoNorm(x, V0, sig=1, mu=0):
    return V0 * np.exp(-((x - mu) / sig) ** 2)


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


def t_choen2(L, V0, E, m=1, hbar=1):
    if E > V0:
        diff = (E - V0)
        arg = np.sqrt(2 * m * diff) * L / hbar
        trans = (4 * E * diff) / (4 * E * diff + V0 ** 2 * np.sin(arg) ** 2)
    if abs(E - V0) < 10 ** (-15):
        diff = 10 ** -99
        arg = np.sqrt(2 * m * diff) * L / hbar
        trans = (4 * E * diff) / (4 * E * diff + V0 ** 2 * np.sin(arg) ** 2)
    return trans


def t_choen(L, V0, E, m=1, hbar=1):
    diff = (E - V0)
    root_eps = np.sqrt(2 * m * diff) * L / hbar
    V_star = V0 ** 2 * L ** 2 / (4 * E * hbar ** 2)
    return 1 / (1 + V_star * np.sinc(root_eps) ** 2)


# hbar = 1.0545718 * 10 ** -34
# m = 1.44316072 * 10 ** -25
# scale = 2 * np.pi * hbar
#
# lim = 4 * 10 ** -5
# N = 2 ** 12
# dx = 2 * lim / N
# x = np.arange(-lim, lim, dx)
#
# # bar_amp = 10 ** 3 * scale
# # E = 10 ** 3 * scale
# L = 4 * 10 ** -6
#
# bar_amp = 100*hbar ** 2 / (m * L ** 2)
# E = bar_amp

m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
L = 4 * 10 ** (-7)
V0 = 10 ** -30
print(V0)

e = V0
l = np.sqrt(hbar ** 2 / (2 * m * V0))
print(l)

L=10*l
E=V0

x = np.linspace(-10, 10, 1000)*l
dx = x[1] - x[0]


hbar = 1
m = 1

lim = 10
N = 2 ** 12
dx = 2 * lim / N
x = np.arange(-lim, lim, dx)

bar_amp = 10
E = 10
L = 4
sig = 2

V_x = barrier(x, bar_amp, -L / 2, L / 2)
# plt.plot(x, V_x)
plt.plot(x,gaussianNoNorm(x,bar_amp,L))
plt.show()

######## Comparison
# n = 1000
# E_range = np.linspace(1, 4, n) * bar_amp
# choen = [t_choen2(L, bar_amp, e, m=m, hbar=hbar) for e in E_range]
# imp = [Impedence(V_x, e, m=m, hbar=hbar) for e in E_range]
# plt.plot(E_range, choen)
# plt.plot(E_range, imp)
# plt.xlabel("Energy")
# plt.ylabel("Transmission Probability")
# plt.show()
#
# plt.plot(E_range, abs(np.asarray(choen) - np.asarray(imp)))
# plt.show()

####### Square Barrier
# n = 1000
# V_list = np.linspace(0.5, 1, n) * bar_amp
# choen_v = [t_choen2(L, v, E, m=m, hbar=hbar) for v in V_list]
# plt.plot(E/V_list, choen_v)
# plt.title("Square Barrier")
# plt.xlabel("V0/E")
# plt.ylabel("Transmission Probability")
# # plt.savefig("Choen_Tunneling")
# plt.show()

####### Same stat width
# square = barrier(x, bar_amp, -L / 2, L / 2)
# sig = L * (1 / (np.sqrt(np.pi) * 6) ** (1 / 3))
# gauss = gaussianNoNorm(x, bar_amp, sig, mu=0)
# plt.plot(x, square)
# plt.plot(x, gauss)
# plt.show()

######## Gaussian
# n = 100
# V_list = np.linspace(0.5, 1.5, n) * V0
#
# # plt.plot(x, gaussianNoNorm(x, bar_amp, L, sig=sig))
# # plt.plot(x, V_x)
# # plt.show()
#
# sig = L * (1 / (np.sqrt(np.pi) * 6) ** (1 / 3))
# T_prob = []
# for v in V_list:
#     V_g = gaussianNoNorm(x, v, sig)
#     imp = Impedence(V_g, E, m=m, hbar=hbar)
#     T_prob.append(imp)
#
# plt.rcParams.update({"font.size":14})
# plt.plot(E / V_list, T_prob)
# plt.xlabel(r"E/$V_0$")
# plt.ylabel("Transmission Probability")
# plt.xlim(min(E / V_list), max(E / V_list))
# plt.ylim(0, 1.04)
# plt.tight_layout()
# plt.savefig("Gaussian Tunneling")
# plt.show()

######## Square Imp
# n = 1000
# V_list = np.linspace(0.5, 1.5, n) * V0
#
# # plt.plot(x, gaussianNoNorm(x, bar_amp, L, sig=sig))
# # plt.plot(x, barrier(x, bar_amp, -L / 2, L / 2))
# # plt.show()
#
# T_prob = []
# for v in V_list:
#     V_g = barrier(x, v, -L / 2, L / 2)
#     imp = Impedence(V_g, E, m=m, hbar=hbar)
#     T_prob.append(imp)
#
# plt.plot(E/V_list, T_prob)
# plt.xlabel(r"E/$V_0$")
# plt.ylabel("Transmission Probability")
# plt.xlim(min(E/V_list),max(E/V_list))
# plt.ylim(0,1.04)
# plt.tight_layout()
# plt.savefig("Choen_Tunneling")
# plt.show()
