from Schrodinger_Solver import Schrodinger
from Animate import Animate
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from scipy.integrate import simps


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


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


def T_t(l, V0,norm=False):
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


def T_array(X_list, Y_list,norm=False):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_t(yv, xv,norm=norm)
    return temp


def contourPlot(V_list, L_list, title=None, show=False,norm=False):
    T_prob = T_array(V_list, L_list,norm=norm)
    plt.figure()
    plt.contourf(V_list / (scale * 10 ** 3), L_list * 10 ** 6, T_prob.clip(min=0), 30, cmap="YlOrBr")
    plt.colorbar()
    plt.xlabel("V0 (kHz)")
    plt.ylabel("Barrier Width (micrometers)")
    if title != None:
        plt.title(title)
        if norm:
            plt.savefig(title+" Norm")
        else:
            plt.savefig(title)
    if show:
        plt.show()
    return 0


w_list = np.asarray([10 ** -8, 10 ** -7, 0.2 * 10 ** -6, 0.3 * 10 ** -6, 0.5 * 10 ** -6, 10 ** -6])
n = 100


V_list = np.linspace(0.5, 1.2, n) * bar_amp
L_list = np.linspace(0.5, 2, n) * L

for i in w_list:
    w = i
    contourPlot(V_list, L_list, title=f"Tanh Tunneling w={i}",norm=True)
    plt.figure()
    plt.title(f"Barrier Shape w={i}")
    plt.plot(x * 10 ** 6, tanhBarrierNorm(x, V_list[0], L_list[0], i) / (scale * 10 ** 3), label="Smallest Barrier")
    plt.plot(x * 10 ** 6, tanhBarrierNorm(x, V_list[0], L_list[-1], i) / (scale * 10 ** 3), label="Largest Barrier")
    plt.xlabel("x (micrometers)")
    plt.ylabel("V (kHz)")
    plt.legend(fancybox="True")
    plt.savefig(f"Barrier Shape w={i} Norm")
