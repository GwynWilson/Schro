from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt

from Input_Parameters_Realistic import *


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


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


def T_s(V0, E):
    v_sq = squareBarrier(x, V0, -L / 2, L / 2)
    k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))
    psi_x = gauss_init(x, k0, x0, d=sig)
    sch = Schrodinger(x, psi_x, v_sq, hbar=hbar, m=m, t=0, args=L / 2)
    return sch.impedence(E)


def T_array(E_list, V_list):
    # E is the x axis, V is the Y
    temp = np.zeros((len(V_list), len(E_list)))
    for i, v in enumerate(V_list):
        print(f"{i}")
        for j, e in enumerate(E_list):
            temp[i, j] = T_s(v, e)
    return temp


E_list = np.linspace(0.5, 5, 200) * bar_amp
V_list = np.linspace(0.5, 5, 200) * bar_amp
T_list = T_array(E_list, V_list)
a = plt.contourf(E_list / scale, V_list / scale, T_list.clip(min=0))
plt.colorbar(a)
plt.xlabel("Energy (Hz)")
plt.ylabel("V0 (Hz)")
plt.title("Square Barrier Transmisson Probablity")
plt.savefig("Square Barrier More")
plt.show()

# x = np.linspace(0,4,3)
# y = np.linspace(0,4,2)
# X, Y = np.meshgrid(x,y)
# Z = np.sqrt(X**2+Y**2)
# print(Z)
# print(T_array(x,y))

# a = plt.contourf(X,Y,Z)
# plt.colorbar(a)
# plt.show()
