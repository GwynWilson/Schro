import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from matplotlib import colors
from scipy.integrate import simps


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


def cos2(x, A, L):
    x = np.asarray(x)
    bar = []
    for i in x:
        if i > -L and i < L:
            bar.append(A * (1 + np.cos(np.pi * i / L)) / 2)
        else:
            bar.append(0)
    return np.asarray(bar)


def gwynCos(x, L_g, par):
    # L_c = L_g * (1.5 * (np.pi ** (5 / 2) / (np.pi ** 2 - 6))) ** (1 / 3)
    # print("xs", L_c / L_g)
    L_c = np.sqrt(0.5*(1/3 - 2/(np.pi**2))**(-1)) * L_g
    # print("Stat", L_c / L_g)
    return par * gauss(x, 1, L_g) + (1 - par) * cos2(x, 1, L_c)


def T_LV(L, V0, w):
    V_x = V0 * gwynCos(x, L, w)
    return Impedence(V_x, E, m=m, hbar=hbar)


def T_arrayLV(X_list, Y_list, w):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"Par={w},Ind={i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_LV(yv, xv, w)
    return temp


def T_arrayVW(X_list, Y_list, L):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        print(f"Par={L},Ind={i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = T_LV(L, xv, yv)
    return temp


####Basic Plots
bar_amp = 10 ** -30
l = np.sqrt(hbar ** 2 / (2 * m * bar_amp))
L = 10 * l

# L_cos = np.sqrt(np.pi / 2) * L
#
# x = np.linspace(-25, 25, 1000) * l
# dx = x[1] - x[0]
#
# g = gwynCos(x, L, 1)
# c = gwynCos(x, L, 0)
# c2 = cos2(x, 1, L_cos)
#
# g_I0 = simps(g, x, dx)
# c_I0 = simps(c, x, dx)
# c2_I0 = simps(c2, x, dx)
#
# print("g_I0", g_I0)
# print("c_I0", c_I0, L_cos)
#
# g_I2 = simps(x ** 2 * np.asarray(g), x, dx)
# c_I2 = simps(x ** 2 * np.asarray(c), x, dx)
# c2_I2 = simps(x ** 2 * np.asarray(c2), x, dx)
#
# print("g_statw", g_I2 / g_I0, L ** 2 / 2)
# print("c_statw", c_I2 / c_I0, L_cos ** 2 / np.pi)
# print("c2_statw", c2_I2 / c2_I0, L_cos ** 2 * (1/3 - 2/np.pi**2))

# plt.rcParams.update({"font.size": 14})
# plt.plot(x / l, gwynCos(x, L, 0), label=r"$0$")
# plt.plot(x / l, gwynCos(x, L, 0.5), label=r"$0.5$")
# plt.plot(x / l, gwynCos(x, L, 1), label=r"$1$")
# plt.xlim(-25, 25)
# plt.ylim(0, 1)
# plt.legend()
# plt.xlabel(r"$x/l$")
# plt.ylabel(r"$V(x)/V_0$")
# plt.tight_layout()
# plt.savefig("gCos_shape")
# plt.show()
#
# plt.figure()
# plt.plot(x / l, gwynCos(x, L, 0), label=r"$0$")
# plt.plot(x / l, gwynCos(x, L, 0.5), label=r"$0.5$")
# plt.plot(x / l, gwynCos(x, L, 1), label=r"$1$")
# plt.xlim(10, 25)
# plt.ylim(0, 0.25)
# plt.legend()
# plt.xlabel(r"$x/l$")
# plt.ylabel(r"$V(x)/V_0$")
# plt.tight_layout()
# plt.savefig("gCos_shape_zoom")
# plt.show()

######Transmission Probability
# n = 100
# E = bar_amp
#
# diff = 0.1
# # V_list = np.linspace(1 / (1 - diff), 1 / (1 + diff), n) * bar_amp
# V_list = np.linspace(0.2, 0.5, n) * bar_amp

# T_list_c = []
# T_list_g = []
# T_list_m = []
# for i in V_list:
#     v_temp_c = i * gwynCos(x, L, 0)
#     v_temp_g = i * gwynCos(x, L, 1)
#     v_temp_m = i * gwynCos(x, L, 0.5)
#     T_list_c.append(Impedence(v_temp_c, E, m=m, hbar=hbar))
#     T_list_g.append(Impedence(v_temp_g, E, m=m, hbar=hbar))
#     T_list_m.append(Impedence(v_temp_m, E, m=m, hbar=hbar))
#
# datarr = np.array([T_list_c, T_list_m, T_list_g])
# np.savetxt("Cos_zoom_4.txt", datarr)

# V_list = np.linspace(1 / (1 - diff), 1 / (1 + diff), n) * bar_amp
# plt.rcParams.update({"font.size": 14})
# datarr = np.loadtxt("Cos_basic_4.txt")
# plt.plot(E / V_list, 1 - datarr[0], label="0")
# plt.plot(E / V_list, 1 - datarr[1], label="0.5")
# plt.plot(E / V_list, 1 - datarr[2], label="1")
# plt.legend(loc=4)
# plt.xlim(1 - diff, 1 + diff)
# plt.ylim(1, 0)
# plt.xlabel(r"E/$V_0$")
# plt.ylabel("Reflection Coefficient")
# plt.tight_layout()
# plt.savefig("gCos_ref")
# plt.show()

# V_list = np.linspace(0.2, 0.5, n) * bar_amp
# plt.rcParams.update({"font.size": 14})
# datarr = np.loadtxt("Cos_zoom_4.txt")
# plt.plot(E / V_list, (1 - datarr[0])/(1e-7), label="0")
# plt.plot(E / V_list, (1 - datarr[1])/(1e-7), label="0.5")
# plt.plot(E / V_list, (1 - datarr[2])/(1e-7), label="1")
# plt.legend()
# plt.xlim(min(E/V_list), max(E/V_list))
# plt.ylim((1-min(datarr[0]))/1e-7,-0.2e-7/1e-7)
# plt.xlabel(r"$V_0/E$")
# plt.ylabel(r"Reflection Coefficient ($1e-7$)")
# plt.tight_layout()
# plt.savefig("gCos_ref_zoom")
# plt.show()

######### 2D w
n = 200
# l = 1E-5
# bar_amp = 100*hbar**2/(m*sig**2)
bar_amp = 10**-30
E= bar_amp

l = np.sqrt(hbar ** 2 / (2 * m * bar_amp))
L = 20 * l

V_list = np.linspace(0.5, 0.75, n) * bar_amp
V_list = np.linspace(1/5, 1/1.5, n) * bar_amp
w_list = np.linspace(0, 1, n)


T_prob = T_arrayVW(V_list, w_list, L)
np.savetxt(f"gCos_warr_{n}.txt", T_prob)

T_prob = 1-np.loadtxt(f"gCos_warr_{n}.txt")
norm = colors.Normalize(vmin=T_prob.min(), vmax=T_prob.max())
print(T_prob.min())

plt.figure()
a = plt.pcolormesh(E/V_list, w_list, T_prob, cmap="gist_heat_r")
cbar = plt.colorbar(a,ticks=[0,1.e-08, 2.e-08, 3.e-08, 4.e-08, 5.e-08, 6.e-08, 7.e-08], format='%.0e')
cbar.ax.invert_yaxis()
z = cbar.get_ticks()
cbar.set_ticklabels(["0","1.e-08", "2.e-08", "3.e-08", "4.e-08", "5.e-08", "6.e-08", "7.e-08"])


plt.xlabel(r"E/$V_0$")
plt.ylabel(r"$\gamma$")
plt.tight_layout()
plt.savefig("Cos_par.png")
plt.show()

###### 2D Plot
# n = 200
# V_list = np.linspace(0.5, 0.75, n) * bar_amp
# L_list = np.linspace(5, 15, n) * sig
# w_list = [0, 0.5, 0.25, 1]
# # w_list=[0.1]
#
# # for par in w_list:
# #     # T_prob = T_arrayLV(V_list, L_list, par)
# #     # np.savetxt(f"gCos_{par}.txt", T_prob)
# #
# #     T_prob=np.loadtxt(f"gCos_{par}.txt")
# #     plt.figure()
# #     a = plt.pcolormesh(V_list / E, L_list/(10**-6), T_prob)
# #     plt.colorbar(a)
# #     plt.xlabel("V0/E")
# #     plt.ylabel("L (micrometers)")
# #     plt.title(f"Interpolating Barrier Spectra w={w}")
# #     # plt.savefig(f"Interpolate{w}.png")
# #     plt.show()
#
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
# fig.subplots_adjust(top=0.91, bottom=0.11, left=0.09, right=0.9, hspace=0.2, wspace=0.2)
# fig.suptitle("Interpolation Tunneling")
#
# dat1 = 1-np.loadtxt(f"gCos_{w_list[0]}.txt")
# norm=colors.Normalize(vmin=dat1.min(),vmax=dat1.max())
# axs[0, 0].pcolormesh(V_list / E, L_list / (10 ** -6), dat1, cmap="gist_heat",norm=norm)
# axs[0, 0].set_title(f"w = {w_list[0]}")
#
# dat2 = 1-np.loadtxt(f"gCos_{w_list[1]}.txt")
# axs[0, 1].pcolormesh(V_list / E, L_list / (10 ** -6), dat2, cmap="gist_heat",norm=norm)
# axs[0, 1].set_title(f"w={w_list[1]}")
#
# dat3 = 1-np.loadtxt(f"gCos_{w_list[2]}.txt")
# axs[1, 0].pcolormesh(V_list / E, L_list / (10 ** -6), dat3, cmap="gist_heat",norm=norm)
# axs[1, 0].set_title(f"w={w_list[2]}")
#
# dat4 = 1-np.loadtxt(f"gCos_{w_list[3]}.txt")
# im = axs[1, 1].pcolormesh(V_list / E, L_list / (10 ** -6), dat4, cmap="gist_heat",norm=norm)
# axs[1, 1].set_title(f"w={w_list[3]}")
#
# fig.text(0.5, 0.05, "V0/E", va='center', ha='center')
# fig.text(0.04, 0.5, "Barrier Length (micrometers)", va='center', ha='center', rotation='vertical')
#
# print(dat1.min(),dat2.min(),dat3.min(),dat4.min())
#
# cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(im, cax=cb_ax)
#
# # plt.savefig("Cos_Subplots.png")
# plt.setp(axs, ylim=[5 * sig / (10 ** -6), 10 * sig / (10 ** -6)])
# plt.show()
