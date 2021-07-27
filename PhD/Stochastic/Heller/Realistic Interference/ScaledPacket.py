from PhD.Stochastic.Heller import MultiplePackets as mul
from PhD.Stochastic.Heller import Heller as hel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps


def hellerPacketVal(x, vals):
    xt, vt, at, gt = vals
    return np.exp(
        (1j / hbar) * at * (x - xt) ** 2 + (1j / hbar) * m * vt * (x - xt) + (
                1j / hbar) * gt)


def plotPhase(t, arr1, arr2, hbar=1):
    gl = arr1[:, 3]
    gl2 = arr2[:, 3]
    plt.plot(t, (gl - gl2) / hbar)
    plt.show()


def plotBasic(t, arr, show=True):
    xl = arr[:, 0]
    vl = arr[:, 1]
    al = arr[:, 2]
    gl = arr[:, 3]

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 9))
    axs[0, 0].plot(t, xl)
    axs[0, 1].plot(t, vl)
    axs[1, 0].plot(t, np.imag(al))
    axs[1, 1].plot(t, gl)

    if show:
        plt.show()
    return 0

def plotBasic2(t,arr1,arr2):
    xl1 = arr1[:, 0]
    vl1 = arr1[:, 1]
    al1 = arr1[:, 2]
    gl1 = arr1[:, 3]

    xl2 = arr2[:, 0]
    vl2 = arr2[:, 1]
    al2 = arr2[:, 2]
    gl2 = arr2[:, 3]

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
    axs[0, 0].plot(t, xl1)
    axs[0, 1].plot(t, vl1)
    axs[1, 0].plot(t, np.imag(al1))
    axs[1, 1].plot(t, gl1)

    axs[0, 0].plot(t, xl2)
    axs[0, 1].plot(t, vl2)
    axs[1, 0].plot(t, np.imag(al2))
    axs[1, 1].plot(t, gl2)

    plt.show()

def derivsdt(t, current, args, eta, dt):
    m, hbar = args
    x = current[1] * dt
    v = 0
    a = (-2 * current[2] ** 2 / m) * dt
    g = 1j * hbar * current[2] * dt / m + m * current[1] ** 2 * dt / 2
    return x, v, a, g


def HOdt(t, current, args, eta, dt):
    m, hbar, w, sig_n = args
    x = current[1] * dt
    v = -w ** 2 * (current[0] * dt - eta * sig_n)
    a = ((-2 * current[2] ** 2 / m) - (m * w ** 2) / 2) * dt
    g = (1j * hbar * current[2] / m + m * current[1] ** 2 / 2) * dt - 0.5 * m * w ** 2 * current[
        0] ** 2 * dt + m * w ** 2 * current[0] * eta * sig_n - 0.5 * m * w ** 2 * sig_n ** 2 * eta ** 2
    return x, v, a, g


def HOexpected(t, args, init):
    t = np.asarray(t)
    m, hbar, w, sig_n = args
    x0, v0, a0, g0 = init
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t)
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))
    g_ex = g0 - hbar * w * t / 2 + 0.5 * m * (v_ex * x_ex - v0 * x0)

    nt = len(t)
    hel_dat = np.zeros((nt - 1, len(init)), dtype=complex)  # Unsure about this -1
    for i in range(nt - 1):  # Unsure about this -1
        hel_dat[i] = [x_ex[i], v_ex[i], a_ex[i], g_ex[i]]

    return hel_dat


############ Actual Values
m_p = 1.44316072 * 10 ** -25
hbar_p = 1.0545718 * 10 ** -34
w_p = 0.1
sig_p = 10 ** -3
T_p = 0.8
Vk_p = 0.01
x0_p = 0
V0_p = 0

########### Scaling so m=1,hbar =1, t scaled by period of oscillator
m_s = m_p
T_s = 2 * np.pi / w_p
L_s = np.sqrt(hbar_p * T_s / m_s)

# m_s = 1
# T_s = 1
# L_s = 1

########## Applying Scalings
N_atom = 1
m = N_atom * m_p / m_s
hbar = hbar_p * T_s / (m_s * L_s ** 2)
w = w_p * T_s
T = T_p / T_s
sig = sig_p / L_s
vkick = Vk_p / L_s * T_s
x0 = x0_p / L_s
V0 = V0_p / L_s * T_s
a0 = 1j * hbar / (4 * sig ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)

Ns = 1000
root = (np.arctan(np.sin(w * T / 2) / (2 - np.cos(w * T / 2)))) / w + T / 2
dt = T / Ns

t_temp = np.asarray([i * dt for i in range(Ns)])

rootwhere = np.abs(t_temp - root).argmin()

print(T, root, vkick)

lim = vkick * T
N = 2 ** 12
dx = 2 * lim / N
x = np.asarray([i * dx - lim for i in range(N)])

init = [x0, V0, a0, g0]
init2 = [x0, -V0, a0, g0]
init_array = np.array([init, init2])

########### Single packet no potential
# args = (m, hbar)
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, derivsdt, args, x)
# hel_arr = Solve_hel.onePacketArr(vkick, init)
#
# inital_psi = hellerPacketVal(x, hel_arr[0])
# mid_psi = hellerPacketVal(x, hel_arr[int(Ns / 2)])
# final_psi = hellerPacketVal(x, hel_arr[-1])
# inital_psis = inital_psi * np.conjugate(inital_psi)
# mid_psis = mid_psi * np.conjugate(mid_psi)
# final_psis = final_psi * np.conjugate(final_psi)
#
# plt.plot(x*L_s, inital_psis)
# plt.plot(x*L_s, mid_psis)
# plt.plot(x*L_s, final_psis)
# plt.show()


############ Classical Simulation
# args = (m, hbar, w, 0)
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, HOdt, args, x)
# hel_arr = Solve_hel.onePacketArr(vkick, init)
#
# inital_psi = hellerPacketVal(x, hel_arr[0])
# mid_psi = hellerPacketVal(x, hel_arr[int(Ns / 2)])
# final_psi = hellerPacketVal(x, hel_arr[-1])
#
# inital_psis = inital_psi * np.conjugate(inital_psi)
# mid_psis = mid_psi * np.conjugate(mid_psi)
# final_psis = final_psi * np.conjugate(final_psi)
#
# root_hel = hel_arr[rootwhere]
# # root_hel[1] +=vkick
# root_psi = hellerPacketVal(x, root_hel)
# root_psis = root_psi * np.conjugate(root_psi)
#
# plt.plot(x, inital_psis)
# plt.plot(x, mid_psis)
# # plt.plot(x, final_psis)
# plt.plot(x, root_psis)
# plt.show()
#
# tot_psi = (inital_psi + root_psi) / np.sqrt(2)
# tot_psis = tot_psi * np.conjugate(tot_psi)
#
# plt.plot(x, tot_psis)
# plt.show()
#
# print(init)
# hel_arr_stat = Solve_hel.onePacketArr(0, init)
# final_psi_stat = hellerPacketVal(x, hel_arr_stat[-1])
# final_psi_stats = final_psi_stat * np.conjugate(final_psi_stat)
#
# plt.plot(x, final_psis)
# plt.plot(x, final_psi_stats)
# plt.show()


#################### Interferometry 3
# args = (m, hbar, w, 0)
# V_x = 0.5 * m * w ** 2 * x ** 2
#
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, HOdt, args, x)
# # psi_comb = Solve_hel.interferometry3(vkick)
# arr1, arr2 = Solve_hel.interferometry3Arr(vkick)
# psi_comb = Solve_hel.arrToPacket(arr1, arr2)
#
# Compare = mul.Comparison(Ns - 1, dt, x, psi_comb, psi_comb)
# Compare.animateArr(psi_comb,save="ClassicHO30")

# tl = [i * dt for i in range(Ns - 1)]
#
# plotBasic(tl,arr1)
# plotBasic(tl,arr2)
# plotPhase(tl, arr1, arr2, hbar=hbar)

############ Interferometry 3 No Potential
# args = (m, hbar)
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, derivsdt, args, x)
# arr1, arr2 = Solve_hel.interferometry3Arr(vkick)
# psi_comb = Solve_hel.arrToPacket(arr1, arr2)

# Compare = mul.Comparison(Ns - 1, dt, x, psi_comb, psi_comb)
# Compare.animateArr(psi_comb)

# tl = [i * dt for i in range(Ns - 1)]
# plotBasic(tl, arr1)
# plotBasic(tl, arr2)
# plotPhase(tl, arr1, arr2, hbar=hbar)


########## Interferometry 3 Stochastic  ##########BIG
# sig_n = 5 * vkick * T
# # sig_n=0
#
# args = (m, hbar, w, sig_n)
# V_x = 0.5 * m * w ** 2 * x ** 2
#
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, HOdt, args, x)
# noise = np.sqrt(dt) * np.random.randn(Ns)
# arr1, arr2 = Solve_hel.interferometry3Arr(vkick, noise=noise, symmetric=True)
#
# psi_comb = Solve_hel.arrToPacket(arr1, arr2)
#
# tl = [i * dt for i in range(Ns)]
# plotBasic2(tl, arr1, arr2)
#
# Compare = mul.Comparison(Ns, dt, x, psi_comb, psi_comb)
# Compare.animateArr(psi_comb,save="SymetricHOStoc")


########## Harmonic Oscillator analytic
# args = (m, hbar, w, 0)
# t = [i * dt for i in range(Ns)]
# init_v = [x0, vkick, a0, g0]
# hel_arr = HOexpected(t, args, init_v)
# for i, v in enumerate(hel_arr):
#     if i % 200 == 0:
#         plt.plot(x, hellerPacketVal(x, hel_arr[i]) * np.conjugate(hellerPacketVal(x, hel_arr[i])))
# plt.plot(x, 0.5 * m * w ** 2 * x ** 2)
# plt.show()


######### Harmonic oscillator time test
#
# def HOSOL(t, x0, v0, w):
#     t=np.asarray(t)
#     xt = x0 * np.cos(w * t) + v0 / w * np.sin(w * t)
#     vt = -w * x0 * np.sin(w * t) + v0 * np.cos(w * t)
#     return xt, vt
#
#
# T_p = 5
# T = T_p/T_s
# Ns = 100000
# dt = T / Ns
#
# t = [i * dt for i in range(Ns)]
# T2 = dt * int(Ns / 2)
# thalf = [i * dt for i in range(int(Ns / 2))]
# xhalf, vhalf = HOSOL(thalf, x0, vkick, w)
#
#
# tsecond = [i * dt + T2 for i in range(int(Ns / 2))]
#
# xsec, vsec = HOSOL(thalf, xhalf[-1], vhalf[-1] - 2 * vkick, w)
#
# ttot = np.concatenate((thalf,tsecond[1:]),axis=None)
# xtot = np.concatenate((xhalf,xsec[1:]))
# vtot = np.concatenate((vhalf,vsec[1:]))
#
# d=(2-np.cos(w*T/2)) / np.sin(w*T/2)
# root = (np.arctan(np.sin(w*T/2)/(2-np.cos(w*T/2))))/w +T/2
#
# plt.plot(ttot,xtot)
# plt.axhline(0,color="r",linestyle=":")
# plt.axvline(root,color="k",linestyle=":")
# plt.show()
#
# # plt.plot(ttot,vtot)
# # # plt.axhline(0,color="r",linestyle=":")
# # # plt.axvline(root,color="k",linestyle=":")
# # plt.axhline(vkick*np.cos(w*T/2)-2*vkick,color="r",linestyle=":")
# # plt.show()
