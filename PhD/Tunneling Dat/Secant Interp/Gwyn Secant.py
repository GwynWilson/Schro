import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *


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


def gauss(x, A, sig):
    return A * np.exp(-x ** 2 / (sig ** 2))


def secant(x, A, L):
    return A / (np.cosh(np.pi * x / (2 * L)))


def gwynSecant(x, A, L, par):
    sig = L * np.sqrt(2)
    return par * gauss(x, A, sig) + (1 - par) * secant(x, A, L)


V0 = bar_amp
l = np.sqrt(hbar ** 2 / (2 * m * V0))
L = 20 * l
x = np.linspace(-4.5 * L, 4.5 * L, 5000)
dx = x[1] - x[0]

####### Basic Plots
# plt.plot(x / l, gwynSecant(x, bar_amp, sig, 0) / V0, label="0")
# plt.plot(x / l, gwynSecant(x, bar_amp, sig, 0.5) / V0, label="0.5")
# plt.plot(x / l, gwynSecant(x, bar_amp, sig, 1) / V0, label="1")
# plt.show()

####### Transmission
n = 1000
E = bar_amp

x = np.linspace(-10 * L, 10 * L, 10000)
dx = x[1] - x[0]
E = bar_amp

V_list = np.linspace(0.5, 0.25, n) * bar_amp

# plt.plot(x/l,gauss(x,V0/0.25,np.sqrt(2)*L)/V0)
# plt.show()

T_list_s = []
T_list_g = []
T_list_m = []
for ind, i in enumerate(V_list):
    if ind % 100 == 0:
        print(ind)
    v_temp_s = gwynSecant(x, i, L, 0)
    v_temp_g = gwynSecant(x, i, L, 1)
    v_temp_m = gwynSecant(x, i, L, 0.5)
    T_list_s.append(Impedence(v_temp_s, E, m=m, hbar=hbar))
    T_list_g.append(Impedence(v_temp_g, E, m=m, hbar=hbar))
    T_list_m.append(Impedence(v_temp_m, E, m=m, hbar=hbar))

datarr = np.array([T_list_s, T_list_m, T_list_g])
np.savetxt("Secant_zoom.txt", datarr)

datarr = np.loadtxt("Secant_zoom.txt")
plt.plot(E / V_list, 1 - datarr[0], label="0")
plt.plot(E / V_list, 1 - datarr[1], label="0.5")
plt.plot(E / V_list, 1 - datarr[2], label="1")
plt.legend(loc=4)
plt.xlabel(r"E/$V_0$")
plt.ylabel("Reflection Coefficient")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.savefig("gSech_zoom")
plt.show()

####Sech Transmission
# n = 1000
# E = bar_amp
#
# V_list = np.linspace(0.5, 0.2, n) * bar_amp
#
# T_list_s = []
# for ind, i in enumerate(V_list):
#     if ind % 100 == 0:
#         print(ind)
#     v_temp_s = secant(x, i, L)
#     T_list_s.append(Impedence(v_temp_s, E, m=m, hbar=hbar))
#
# plt.plot(E / V_list, 1 - np.asarray(T_list_s))
# plt.xlabel(r"E/$V_0$")
# plt.ylabel("Reflection Coefficient")
# plt.gca().invert_yaxis()
# # plt.yscale("log")
# plt.tight_layout()
# plt.savefig("Sech_oscill_no_scale")
# plt.show()


####Gauss Transmission
# n = 1000
# x = np.linspace(-10 * L, 10 * L, 10000)
# dx = x[1] - x[0]
# E = bar_amp
#
# V_list = np.linspace(0.5, 0.2, n) * bar_amp
#
# # bar = gauss(x, E/0.25, np.sqrt(2) * L)*(0.25/E)
# # bar = np.asarray(bar) - bar[0]
# # plt.plot(x/l, bar)
# # plt.show()
#
# T_list_s = []
# for ind, i in enumerate(V_list):
#     if ind % 100 == 0:
#         print(ind)
#     v_temp_s = gauss(x, i, np.sqrt(2) * L)
#     # v_temp_s = np.asarray(v_temp_s) - v_temp_s[0] # Setting the ends to zero
#     T_list_s.append(Impedence(v_temp_s, E, m=m, hbar=hbar))
#
# plt.plot(E / V_list, 1 - np.asarray(T_list_s))
# plt.xlabel(r"E/$V_0$")
# plt.ylabel("Reflection Coefficient")
# plt.gca().invert_yaxis()
# # plt.yscale("log")
# plt.tight_layout()
# plt.savefig("Gauss_x_lim_adj")
# plt.show()
