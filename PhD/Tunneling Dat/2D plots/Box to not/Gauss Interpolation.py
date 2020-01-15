from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from scipy.integrate import simps
from scipy.optimize import curve_fit


def gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def barrier(x, w, sig):
    return np.tanh((np.exp(-x ** 2 / (2 * sig ** 2))) / w) * 1 / np.tanh((np.exp(-1 / (2 * sig ** 2))) / w)


def barrier(x, V0, w, sig):
    raw = np.tanh((np.exp(-x ** 2 / (2 * sig ** 2))) / w)
    return V0 * raw / max(raw)


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


def T_b(w, V0, norm=False):
    V_x = barrier(x, V0, w, sig)
    return Impedence(V_x, E, m=m, hbar=hbar)


def T_array(X_list, Y_list, norm=False):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_b(yv, xv, norm=norm)
    return temp


w_list = np.logspace(-5, 1, 100)
V_list = np.linspace(0.5, 1.5, 1000) * bar_amp

######### Comparison
# y_dat = barrier(x, bar_amp, 10, sig)
# popt, covt = curve_fit(gauss, x, y_dat, (bar_amp, 0, sig))
# plt.plot(x, y_dat)
# plt.plot(x, gauss(x, popt[0], popt[1], popt[2]))
# plt.show()


####### Barriers
# for i in w_list:
#     plt.plot(x, barrier(x, bar_amp, i, sig))
# plt.show()

######### 1D plot
# for w in w_list:
#     T_prob = []
#     for i in V_list:
#         V_x = barrier(x, i, w, sig)
#         imp = Impedence(V_x, E, m=m, hbar=hbar)
#         T_prob.append(imp)
#     plt.plot(V_list/E, T_prob,label = f"{w}")
# plt.legend()
# plt.show()


########## 2D
T_porbability = T_array(V_list, w_list)
np.savetxt("Interparray.txt",T_porbability)
for i in T_porbability:
    plt.plot(V_list / E, i)
    plt.show()

a = plt.pcolormesh(V_list / E, w_list, T_porbability)
plt.colorbar(a)
plt.yscale("log")
plt.show()
