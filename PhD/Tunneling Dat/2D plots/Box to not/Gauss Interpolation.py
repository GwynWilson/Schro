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


def barrier_i(x, V0, L, w):
    return V0 * np.tanh(((2 / w + np.e) ** (-4 * x ** 2 / L ** 2)) / w) / np.tanh(1 / w)


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


def T_i(w, V0, norm=False):
    V_x = barrier_i(x, V0, 2 * L, w)
    return Impedence(V_x, E, m=m, hbar=hbar)


def T_LV(L, V0, w):
    V_x = barrier_i(x, V0, L, w)
    return Impedence(V_x, E, m=m, hbar=hbar)


def T_array(X_list, Y_list, norm=False):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_i(yv, xv, norm=norm)
    return temp


def T_arrayLV(X_list, Y_list, w, norm=False):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_LV(yv, xv, w)
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
# w_list = [10 ** -100, 0.1, 100]
# for i in w_list:
#     plt.plot(x/10**-6, barrier(x, bar_amp, L, i)/scale ,label=f"w={i}")
# plt.legend(framealpha=1)
# plt.title("Interpolating Barriers")
# plt.ylabel("Barrier Height (hz)")
# plt.xlabel("x (micrometers)")
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


# ########## 2D
# w_list = np.logspace(-10, 3, 100)
# V_list = np.linspace(0.5, 1.5, 1000) * bar_amp
# T_porbability = T_array(V_list, w_list)
# np.savetxt("Correct_2L.txt", T_porbability)
# # for i in T_porbability:
# #     plt.plot(V_list / E, i)
# #     plt.show()
#
# a = plt.pcolormesh(V_list / E, w_list, T_porbability)
# plt.colorbar(a)
# plt.yscale("log")
# plt.show()

########### Loading Dat
# T_probability = np.loadtxt("Correct_2L.txt")
# a = plt.pcolormesh(V_list / E, w_list, T_probability, cmap="gist_heat")
# # a = plt.contourf(V_list / E, w_list, T_probability, 30, cmap="gist_heat")
# plt.colorbar(a)
# plt.yscale("log")
# plt.xlabel("V0/E")
# plt.ylabel("log(w)")
# plt.title("Transmission Probability For Interpolated Barrier")
# plt.savefig("Gaussian_Interpolation_Nice")
# plt.show()


######## VL plots
# n = 200
# w_list = [0.5]
# V_list = np.linspace(0.5, 1.2, n) * bar_amp
# L_list = np.linspace(0.5, 2, n) * L
# for w in w_list:
#     T_porbability = T_arrayLV(V_list, L_list,w)
#     np.savetxt(f"Correct_{w}.txt", T_porbability)
#
#     plt.figure()
#     a = plt.pcolormesh(V_list / E, L_list/(10**-6), T_porbability)
#     plt.colorbar(a)
#     plt.xlabel("V0/E")
#     plt.ylabel("L (micrometers)")
#     plt.title(f"Interpolating Barrier Spectra w={w}")
#     plt.savefig(f"Interpolate{w}.png")

####### VL data
n = 200
V_list = np.linspace(0.5, 1.2, n) * bar_amp
L_list = np.linspace(0.5, 2, n) * L
w_list = [10 ** -100, 0.01, 0.1, 100]
# for i in load_list:
#     T_probability = np.loadtxt(f"Interpolation/Correct_{i}.txt")
#     a = plt.contourf(V_list / E, L_list/(10**-6), T_probability, 30, cmap="gist_heat")
#     plt.colorbar(a)
#     plt.xlabel("V0/E")
#     plt.ylabel("L (micrometers)")
#     plt.title("Transmission Probability For Interpolated Barrier")
#     # plt.savefig("Gaussian_Interpolation_Nice")
#     plt.show()

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
fig.subplots_adjust(top=0.91, bottom=0.11, left=0.09, right=0.9, hspace=0.2, wspace=0.2)
fig.suptitle("Interpolation Tunneling")

dat1 = np.loadtxt(f"Interpolation/Correct_{w_list[0]}.txt")
axs[0, 0].pcolormesh(V_list / E, L_list / (10 ** -6), dat1, cmap="gist_heat")
axs[0, 0].set_title(f"w = {w_list[0]}")

dat2 = np.loadtxt(f"Interpolation/Correct_{w_list[1]}.txt")
axs[0, 1].pcolormesh(V_list / E, L_list / (10 ** -6), dat2, cmap="gist_heat")
axs[0, 1].set_title(f"w={w_list[1]}")

dat3 = np.loadtxt(f"Interpolation/Correct_{w_list[2]}.txt")
axs[1, 0].pcolormesh(V_list / E, L_list / (10 ** -6), dat3, cmap="gist_heat")
axs[1, 0].set_title(f"w={w_list[2]}")

dat4 = np.loadtxt(f"Interpolation/Correct_{w_list[3]}.txt")
im = axs[1, 1].pcolormesh(V_list / E, L_list / (10 ** -6), dat4, cmap="gist_heat")
axs[1, 1].set_title(f"w={w_list[3]}")

fig.text(0.5, 0.05, "V0/E", va='center', ha='center')
fig.text(0.04, 0.5, "Barrier Length (micrometers)", va='center', ha='center', rotation='vertical')

cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)

plt.savefig("Interpolation_Subplots.png")
plt.show()
