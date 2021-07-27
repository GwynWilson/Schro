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
    v = -w ** 2 * current[0] * dt
    a2 = (-2 * current[2] ** 2 / m) * dt
    a = (-2 * current[2] ** 2 / m) * dt - (m * w ** 2) * dt / 2
    g = 1j * hbar * current[2] * dt / m + m * current[1] ** 2 * dt / 2 - m * w ** 2 * current[0] ** 2 * dt / 2
    return x, v, a, g


m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34

N_atom = 10 ** 6
M = m * N_atom
args = (M, hbar)

T = 1
Ns = 3000
dt = T / Ns

vkivk = 0.01
sig = 10 ** -3

lim = 0.01
N = 2 ** 12
dx = 2 * lim / N
# x = np.arange(-lim, lim, dx)
x = np.asarray([i * dx - lim for i in range(N)])

x0 = 0
vel0 = 0
a0 = 1j * hbar / (4 * sig ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)

init = [x0, vel0, a0, g0]
init2 = [x0, -vel0, a0, g0]
init_array = np.array([init, init2])

######### One packet test
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, derivsdt, args, x)
# hel_arr = Solve_hel.onePacketArr(vkivk, init)
#
# inital_psi = hellerPacketVal(x,hel_arr[0])
# final_psi = hellerPacketVal(x,hel_arr[-1])
# inital_psis = inital_psi*np.conjugate(inital_psi)
# final_psis = final_psi*np.conjugate(final_psi)
#
#
# plt.plot(x,inital_psis)
# plt.plot(x,final_psis)
# plt.show()


w = 50
sig_n = 0

T = np.pi / w
T = 1
print(T)
Ns = 3000
dt = T / Ns

x0 = 0
vel0 = 0
a0 = 1j * hbar / (4 * sig ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)

init = [x0, vel0, a0, g0]
init2 = [x0, -vel0, a0, g0]
init_array = np.array([init, init2])

args = (m, hbar, w, sig_n)

########## Exact Solution
t = [i * dt for i in range(Ns)]
init_v = [x0, vkivk, a0, g0]
hel_arr = HOexpected(t, args, init_v)
for i, v in enumerate(hel_arr):
    if i % 200 == 0:
        plt.plot(x, hellerPacketVal(x, hel_arr[i]) * np.conjugate(hellerPacketVal(x, hel_arr[i])))
plt.plot(x, 0.5 * m * w ** 2 * x ** 2)
plt.show()

###### Packet Not working
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, HOdt, args, x)
# hel_arr = Solve_hel.onePacketArr(vkivk, init)
# for i,v in enumerate(hel_arr):
#     if i%200==0:
#         plt.plot(x, hellerPacketVal(x, hel_arr[i])*np.conjugate(hellerPacketVal(x, hel_arr[i])))
# plt.plot(x,0.5 *m*w**2*x**2)
# plt.show()

##### Sim not working
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, HOdt, args, x)
# hel_arr = Solve_hel.onePacketArr(vkivk, init)
# inital_psi = hellerPacketVal(x, hel_arr[0])
# final_psi = hellerPacketVal(x, hel_arr[-1])
# inital_psis = inital_psi * np.conjugate(inital_psi)
# final_psis = final_psi * np.conjugate(final_psi)
#
# plt.plot(x, inital_psis)
# plt.plot(x, final_psis)
# plt.show()
