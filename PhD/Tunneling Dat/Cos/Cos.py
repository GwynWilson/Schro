from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *


def cosineBarrier(x, L, V0):
    out = 0
    if abs(x) < L / 2:
        out = (V0 / 2) * (1 + np.cos(2 * np.pi * x / L))
    return out


def gaussian(x, L, V0):
    return V0 * np.exp(-4 * x ** 2 / L ** 2)


def tanhBarrier(x, L, A, w):
    raw = np.tanh((x + L / 2) / w) - np.tanh((x - L / 2) / w)
    return A * raw / max(raw)


def varyV(V_list, gauss=False):
    temp = []
    for v in V_list:
        print(v / bar_amp)
        if gauss:
            cb = [gaussian(i, L, v) for i in x]
        else:
            cb = [cosineBarrier(i, L, v) for i in x]
        Sch = Schrodinger(x, x, cb, hbar=hbar, m=m)
        temp.append(Sch.impedence(E=E))
    return temp


def varyE(E_list):
    temp = []
    for e in E_list:
        print(e / 10 / bar_amp)
        cb = [cosineBarrier(i, L, bar_amp) for i in x]
        Sch = Schrodinger(x, x, cb, hbar=hbar, m=m)
        temp.append(Sch.impedence(E=e))
    return temp


###### cosbarrier
# cb = [cosineBarrier(i, L, bar_amp) for i in x]
# plt.plot(x, cb)
# plt.show()

###### Vvaryation
# V_list = np.linspace(0.5, 0.75, 1000) * bar_amp
# T_list = varyV(V_list)
# plt.plot(V_list / E, T_list)
# plt.title("Cos Tunneling")
# plt.xlabel("V0/E")
# plt.ylabel("Tunneling Probability")
# plt.savefig("cos_oscillation.png")
# plt.show()

####### Evaryation
# E_list = np.linspace(1.3,1.5,1000)*E
# T_list = varyE(E_list)
# plt.plot(E_list / bar_amp, T_list)
# plt.savefig("cos_comparison.png")
# plt.show()

####### Gauss Oscillations
V_list = np.linspace(0.5, 0.65, 1000) * bar_amp
T_list = varyV(V_list, gauss=True)
plt.plot(V_list / E, T_list)
plt.title("Gaussian Tunneling")
plt.savefig("gauss.png")
plt.show()

##### Tanh Barrier
# plt.plot(x, tanhBarrier(x, L, bar_amp, 10 ** (-5)))
# plt.show()

# V_list = np.linspace(0.5, 0.87, 1000) * bar_amp
# temp = []
# for v in V_list:
#     cb = tanhBarrier(x, L, v, 10 ** (-5))
#     Sch = Schrodinger(x, x, cb, hbar=hbar, m=m)
#     temp.append(Sch.impedence(E=E))
#
# plt.plot(V_list/E,temp)
# plt.title("Tanh Oscillation")
# plt.xlabel("V0/E")
# plt.ylabel("Tunneling Probability")
# plt.savefig("Tanh_oscillations")
# plt.show()