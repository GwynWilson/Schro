from Schrodinger_Solver import Schrodinger
from Animate import Animate
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


def gaussianBarrier(x, x0, w, A):
    return A * (np.exp(-(x - x0) ** 2 / w ** 2))


def T_s(V0, E, gauss=False):
    if gauss:
        v = gaussianBarrier(x, 0, w, V0)
    else:
        v = squareBarrier(x, V0, -L / 2, L / 2)

    k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))
    psi_x = gauss_init(x, k0, x0, d=sig)
    sch = Schrodinger(x, psi_x, v, hbar=hbar, m=m, t=0, args=L / 2)
    return sch.impedence(E)


def T_g(V0, omeg):
    v = gaussianBarrier(x, 0, omeg, V0)
    E = scale * 10 ** 3
    k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))
    psi_x = gauss_init(x, k0, x0, d=sig)
    sch = Schrodinger(x, psi_x, v, hbar=hbar, m=m, t=0, args=L / 2)
    return sch.impedence(E=E)


def T_array(E_list, V_list, gauss=False):
    # E is the x axis, V is the Y
    temp = np.zeros((len(V_list), len(E_list)))
    for i, v in enumerate(V_list):
        print(f"{i}")
        for j, e in enumerate(E_list):
            temp[i, j] = T_s(v, e, gauss)
    return temp


def contourPlot(E_list, V_list, gauss=False):
    T_list = T_array(E_list, V_list, gauss=gauss)
    plt.figure()
    a = plt.contourf(E_list / scale, V_list / scale, T_list.clip(min=0))
    plt.colorbar(a)
    plt.xlabel("Energy (Hz)")
    plt.ylabel("V0 (Hz)")
    if gauss:
        plt.title("Gaussian Transmisson Probablity")
        plt.savefig("Gaussian Barrier")
    else:
        plt.title("Square Barrier Transmission Pobability")
        plt.savefig("Square Barrier Most")
    plt.show()


def gaussPlot(V_list, w_List):
    T_array = np.zeros((len(V_list), len(w_List)))
    for i, v in enumerate(V_list):
        print(f"{i}")
        for j, omeg in enumerate(w_List):
            T_array[i, j] = T_g(v, omeg)
    plt.figure()
    a = plt.contourf(w_List / 10 ** -6, V_list / scale, T_array.clip(min=0))
    plt.colorbar(a)
    plt.ylabel("Omega (10^6)")
    plt.xlabel("V0 (Hz)")
    plt.title("Gaussian Transmisson Probablity Fixed E")
    plt.savefig("Gaussian Width")
    plt.show()


n = 500

E_list = np.linspace(0.5, 5, n) * bar_amp
V_list = np.linspace(0.5, 5, n) * bar_amp

contourPlot(E_list, V_list)

# V_list = np.linspace(0.5, 1.5, n) * E
# w_list = np.linspace(0.5, 1.5, n) * w
# gaussPlot(V_list, w_list)

# v = gaussianBarrier(x, 0, w, bar_amp)
# E = 2*scale * 10 ** 3
# k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))
# psi_x = gauss_init(x, k0, x0, d=sig)
# sch = Schrodinger(x, psi_x, v, hbar=hbar, m=m, t=0, args=0)
# plt.plot(sch.x, sch.v)
# plt.show()
#
# a = Animate(sch, sch.v, step, dt,lim1=((sch.x[0],sch.x[-1]),(0,max(sch.mod_square_x(r=True)))))
# a.make_fig()

# T_list = []
# for i in E_list:
#     T_list.append(sch.impedence(i))
# plt.plot(E_list,T_list)
# plt.show()


# T_list = T_array(E_list, V_list)
# a = plt.contourf(E_list / scale, V_list / scale, T_list.clip(min=0))
# plt.colorbar(a)
# plt.xlabel("Energy (Hz)")
# plt.ylabel("V0 (Hz)")
# plt.title("Square Barrier Transmisson Probablity")
# plt.savefig("Square Barrier More")
# plt.show()

# x = np.linspace(0,4,3)
# y = np.linspace(0,4,2)
# X, Y = np.meshgrid(x,y)
# Z = np.sqrt(X**2+Y**2)
# print(Z)
# print(T_array(x,y))

# a = plt.contourf(X,Y,Z)
# plt.colorbar(a)
# plt.show()
