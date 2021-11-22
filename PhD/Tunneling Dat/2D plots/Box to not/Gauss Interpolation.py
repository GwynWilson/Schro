from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from scipy.integrate import simps
from scipy.optimize import curve_fit


def gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def gauss2(x, sig):
    return bar_amp * np.exp(-0.5 * (x / sig) ** 2)


def barrier(x, w, sig):
    return np.tanh((np.exp(-x ** 2 / (2 * sig ** 2))) / w) * 1 / np.tanh((np.exp(-1 / (2 * sig ** 2))) / w)


def barrier(x, V0, w, sig):
    raw = np.tanh((np.exp(-x ** 2 / (2 * sig ** 2))) / w)
    return V0 * raw / max(raw)


def barrier_i(x, V0, L, w):
    return V0 * np.tanh(((2 / w + np.e) ** (-4 * x ** 2 / L ** 2)) / w) / np.tanh(1 / w)


def barrier2(x, V0, sig, b):
    # return V0 * np.tanh(((b + np.e) ** (-x ** 2 / (sig ** 2)))*b) / np.tanh(b)
    return V0 * np.tanh(((b + np.e) ** (-x ** 2 / (sig ** 2))) * b) / np.tanh(b)


def secant(x, sig):
    return 1 / np.cosh(np.pi * x / (2 * sig))


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
# n = 200
# V_list = np.linspace(0.5, 1.2, n) * bar_amp
# L_list = np.linspace(0.5, 2, n) * L
#
# l = np.sqrt(hbar ** 2 / (2 * m * bar_amp))
#
# w_list = [10 ** -100, 0.01, 0.1, 100]
# # for i in load_list:
# #     T_probability = np.loadtxt(f"Interpolation/Correct_{i}.txt")
# #     a = plt.contourf(V_list / E, L_list/(10**-6), T_probability, 30, cmap="gist_heat")
# #     plt.colorbar(a)
# #     plt.xlabel("V0/E")
# #     plt.ylabel("L (micrometers)")
# #     plt.title("Transmission Probability For Interpolated Barrier")
# #     # plt.savefig("Gaussian_Interpolation_Nice")
# #     plt.show()
#
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
# fig.subplots_adjust(top=0.91, bottom=0.11, left=0.09, right=0.9, hspace=0.2, wspace=0.2)
# # fig.suptitle("Interpolation Tunneling")
#
# dat1 = np.loadtxt(f"Interpolation/Correct_{w_list[0]}.txt")
# axs[0, 0].pcolormesh(E / V_list, L_list / l, dat1, cmap="gist_heat")
# axs[0, 0].set_title(fr"$\alpha$ = {1/w_list[0]}")
# axs[0, 0].set_ylim((10, 30))
#
# dat2 = np.loadtxt(f"Interpolation/Correct_{w_list[1]}.txt")
# axs[0, 1].pcolormesh(E / V_list, L_list / l, dat2, cmap="gist_heat")
# axs[0, 1].set_title(fr"$\alpha$ = {1/w_list[1]}")
#
# dat3 = np.loadtxt(f"Interpolation/Correct_{w_list[2]}.txt")
# axs[1, 0].pcolormesh(E / V_list, L_list / l, dat3, cmap="gist_heat")
# axs[1, 0].set_title(fr"$\alpha$ = {1/w_list[2]}")
#
# dat4 = np.loadtxt(f"Interpolation/Correct_{w_list[3]}.txt")
# im = axs[1, 1].pcolormesh(E / V_list, L_list / l, dat4, cmap="gist_heat")
# axs[1, 1].set_title(fr"$\alpha$ = {1/w_list[3]}")
#
# fig.text(0.5, 0.05, r"E/$V_0$", va='center', ha='center')
# fig.text(0.04, 0.5, r"$L_I/l$", va='center', ha='center', rotation='vertical')
#
# cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(im, cax=cb_ax)
#
# plt.savefig("Interpolation_Subplots.png")
# plt.show()

###### New barrier test
# y_dat = barrier2(x, bar_amp, sig, 10000)
# y_dat2 = barrier2(x, bar_amp, sig, 0.00001)
# plt.plot(x/10**-6,y_dat/scale)
# plt.plot(x / 10 ** -6, y_dat2 / scale)
# plt.xlim(-10, 10)
# plt.show()


####### Secant
sig=3*sig
print(sig)
bar_amp=100*hbar**2/(m*sig**2)


bar_amp = 10**-30
m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
V0 = 10 ** -30

e = bar_amp
l = np.sqrt(hbar ** 2 / (2 * m * V0))
print(l)
sig = 5 * l

x = np.linspace(-20, 20, 1000) * l
dx = x[1] - x[0]

E=bar_amp
y_dat = bar_amp*secant(x,sig)
y_dat2 = bar_amp*(secant(x,4*sig)**10)
# y_dat3 = bar_amp*(secant(x,1300*sig)**1000000/1000)

popt,covt = curve_fit(gauss2,x,y_dat2)


# plt.plot(x/l,y_dat/bar_amp,label="Sech")
# plt.plot(x/l,y_dat2/bar_amp,label=r"Sech$^{10}$")
# # plt.plot(x/sig,y_dat3/scale,label=r"Sech$^{1000000}$",linestyle="--")
# # plt.plot(x/sig,gauss2(x,0.83*sig)/bar_amp, label="Gaussian",linestyle="--")
# # plt.title("Sech Barrier")
# plt.xlabel(r"$x/l$")
# plt.ylabel(r"$V(x)/V_0$")
# plt.legend()
# plt.xlim((-20,20))
# plt.ylim((0,1.05))
# plt.tight_layout()
# plt.show()

# plt.rcParams.update({"font.size":14})
# V_list = np.linspace(1, 0.5, 100) * bar_amp
# T_prob = []
# for i in V_list:
#     bar = i*secant(x,sig)
#     imp = Impedence(bar, E, m=m, hbar=hbar)
#     T_prob.append(imp)
# plt.plot(E/V_list[::], T_prob[::])
# plt.title("Sech Barrier Transmission")
# plt.xlabel("E/V0")
# plt.ylabel("Transmission Coeficcient")
# plt.ylim((min(T_prob),max(T_prob)+0.02))
# plt.xlim((min(E/V_list),max(E/V_list)))
# plt.show()

plt.rcParams.update({"font.size":14})
# V_list = np.linspace(0.5, 1.5, 100) * bar_amp
V_list = np.linspace(1/2.5, 1/1.5, 100) * bar_amp
T_prob_sec = []
T_prob_pow = []
for i in V_list:
    bar = i*(secant(x,4*sig)**10)
    bar2 = i * secant(x, sig)
    # plt.plot(x,bar)
    # plt.plot(x,bar2)
    # plt.show()
    imp1 = Impedence(bar, E, m=m, hbar=hbar)
    T_prob_pow.append(imp1)
    imp2 = Impedence(bar2, E, m=m, hbar=hbar)
    T_prob_sec.append(imp2)
plt.plot(E/V_list, T_prob_sec,label="Sech")
plt.plot(E/V_list, T_prob_pow,label=r"Sech$^{10}$")
# plt.title("Sech Barrier Transmission")
plt.xlabel(r"$E/V_0$")
plt.ylabel("Transmission Probability")
plt.legend(loc=4)
plt.ylim((min(T_prob_sec),max(T_prob_sec)+0.0002))
plt.xlim((min(E/V_list),max(E/V_list)))
plt.tight_layout()
plt.show()
