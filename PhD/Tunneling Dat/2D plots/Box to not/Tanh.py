from Schrodinger_Solver import Schrodinger
from Animate import Animate
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from scipy.integrate import simps


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


def gaussian(x, sig=1, mu=0):
    return 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def tanhBarrier(x, A, L, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    return A * raw / max(raw)


def tanhBarrierNorm(x, A, L, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    norm = simps(raw, x)
    raw = raw / norm
    raw *= L * A
    return raw


def tanhBarrieNoNormr(x, A, L, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    return A * raw / 2


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


def tanhGauss(x, w, A, L, sig=1, mu=0):
    raw = A * np.tanh((gaussian(x, sig=sig, mu=mu)) / w)
    return raw


def T_t(l, V0, norm=False):
    if norm:
        v = tanhBarrierNorm(x, V0, l, w)
    else:
        v = tanhBarrier(x, V0, l, w)
    psi_x = gauss_init(x, k0, x0, d=sig)
    sch = Schrodinger(x, psi_x, v, hbar=hbar, m=m, t=0)
    # plt.plot(x,v/scale)
    # plt.show()
    return sch.impedence(E)


def T_s(l, V0):
    v = squareBarrier(x, V0, -l / 2, l / 2)
    psi_x = gauss_init(x, k0, x0, d=sig)
    sch = Schrodinger(x, psi_x, v, hbar=hbar, m=m, t=0)
    return sch.impedence(E)


def T_array(X_list, Y_list, norm=False):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_t(yv, xv, norm=norm)
    return temp


def contourPlot(V_list, L_list, title=None, show=False, norm=False, dat=False):
    T_prob = T_array(V_list, L_list, norm=norm)
    plt.figure()
    plt.contourf(V_list / (scale * 10 ** 3), L_list * 10 ** 6, T_prob.clip(min=0), 30, cmap="YlOrBr")
    plt.colorbar()
    plt.xlabel("V0 (kHz)")
    plt.ylabel("Barrier Width (micrometers)")
    if title != None:
        plt.title(title)
        if norm:
            plt.savefig(title + " Norm")
        else:
            plt.savefig(title)
        if dat:
            np.savetxt(f"tanh_{title}.txt", T_prob)
    if show:
        plt.show()

    return 0


w_list = np.asarray([10 ** -8, 10 ** -7, 0.2 * 10 ** -6, 0.3 * 10 ** -6, 0.5 * 10 ** -6, 10 ** -6])
n = 100

w_transform = np.logspace(1, 6, num=2)
w = 10 ** 6
sig = 10 ** -6
L_list = np.linspace(0.5, 2, 2) * L
s_list = np.logspace(-6, -5.5, 2)

# for s in s_list:
#         plt.plot(x, tanhGauss(x, w_transform[0], bar_amp, L, sig=s))
# for wv in w_transform:
#     plt.plot(x, tanhGauss(x, wv, bar_amp, L, sig=sig))
# plt.show()

V_list = np.linspace(0.5, 1.2, 100) * bar_amp
w_transform = np.logspace(1, 5.5, num=2)
V = tanhGauss(x, 10, bar_amp, L, sig=sig)
s_list = np.logspace(-6, -5.5, 100)

####New Method attempt
# T_prob = np.zeros((len(V_list), len(s_list)))
# for i, v0 in enumerate(V_list):
#     print(f"{i}")
#     for j, s in enumerate(s_list):
#         V = tanhGauss(x, w_transform[0], v0, L, sig=s)
#         psi_x = gauss_init(x, k0, x0, d=sig)
#         sch = Schrodinger(x, psi_x, V, hbar=hbar, m=m, t=0)
#         T_prob[i, j] = sch.impedence(E)
#
# plt.figure()
# plt.contourf(V_list / (scale * 10 ** 3), s_list, T_prob.clip(min=0), 30, cmap="YlOrBr")
# plt.colorbar()
# plt.show()


######## 2D Spectra
# n = 200
# V_list = np.linspace(0.5, 1.2, n) * bar_amp
# L_list = np.linspace(0.5, 2, n) * L
#
# dat = True
#
# if dat:
#     np.savetxt("tanh_V_list.txt", V_list)
#     np.savetxt("tanh_L_list.txt", L_list)
#     np.savetxt("tanh_w_list.txt", w_list)
#
# for i in w_list:
#     w = i
#     contourPlot(V_list, L_list, title=f"Tunneling w={w}", dat=dat)
#     plt.figure()
#     plt.title(f"Barrier Shape w={i}")
#     plt.plot(x * 10 ** 6, tanhBarrierNorm(x, V_list[0], L_list[0], i) / (scale * 10 ** 3), label="Smallest Barrier")
#     plt.plot(x * 10 ** 6, tanhBarrierNorm(x, V_list[0], L_list[-1], i) / (scale * 10 ** 3), label="Largest Barrier")
#     plt.xlabel("x (micrometers)")
#     plt.ylabel("V (kHz)")
#     plt.legend(fancybox="True")
#     plt.savefig(f"Barrier Shape w={i} Norm")

########## Changing L
# n = 1000
# V0 = 0.5 * bar_amp
# L_list = np.linspace(0.5, 2, n) * L
# w_list = np.asarray([10 ** -8, 10 ** -7, 0.2 * 10 ** -6, 0.3 * 10 ** -6, 10 ** -6])
# for w in w_list:
#     T_prob = []
#     for l in L_list:
#         T_prob.append(T_t(l, V0))
#     plt.plot(L_list / 10 ** (-6), T_prob, label=f"w={w}")
#
# plt.title("Tunneling Spectra fixed V0/E=0.5")
# plt.xlabel("a (micrometers)")
# plt.ylabel("Tunneling Probability")
# plt.legend(loc=4, framealpha=1)
# plt.savefig("Tanh_a")
# plt.show()

########## Changing E
n = 1000
V_list = np.linspace(0.5, 1.2, n) * bar_amp
w_list = np.asarray([10 ** -8, 0.2 * 10 ** -6, 0.3 * 10 ** -6, 10 ** -6])
for w in w_list:
    T_prob = []
    for v in V_list:
        T_prob.append(T_t(L, v))
    plt.plot(V_list / (scale * 10 ** 3), T_prob, label=f"w={w}")

plt.title(f"Tunneling Spectra fixed L={L}m")
plt.xlabel("V0/E")
plt.ylabel("Tunneling Probability")
plt.legend(loc="best", framealpha=1)
plt.savefig("Tanh_W_Tunneling.png")
plt.show()
