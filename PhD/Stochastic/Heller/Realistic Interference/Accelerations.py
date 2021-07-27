from PhD.Stochastic.Heller import MultiplePackets as mul
from PhD.Stochastic.Heller import Heller as hel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps


def derivsdt(t, current, args, eta, dt):
    m, hbar, gr = args
    x = current[1] * dt
    v = -gr * dt
    a = (-2 * current[2] ** 2 / m) * dt
    g = 1j * hbar * current[2] * dt / m + m * current[1] ** 2 * dt / 2 - m * gr * current[0] * dt
    return x, v, a, g


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


def plotBasic2(t, arr1, arr2):
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
    axs[1, 0].plot(t, np.imag(al1) / np.imag(a0))
    axs[1, 1].plot(t, gl1)

    axs[0, 0].plot(t, xl2, linestyle="--")
    axs[0, 1].plot(t, vl2, linestyle="--")
    axs[1, 0].plot(t, np.imag(al2) / np.imag(a0), linestyle="--")
    axs[1, 1].plot(t, gl2, linestyle="--")

    axs[0, 0].set_ylabel("x")
    axs[0, 1].set_ylabel("v")
    axs[1, 0].set_ylabel(r"$\alpha/alpha_0$ (imaginary)")
    axs[1, 1].set_ylabel(r"$\gamma$")

    plt.show()


def trajectory(t, x0, v0, v1, v3, gr0):
    t = np.asarray(t)
    traj = np.zeros(len(t))
    for n, i in enumerate(t):
        if n < int(len(t) / 2):
            traj[n] = x0 + v0 * i - 0.5 * gr0 * i ** 2

        # elif n == len(t)-1:
        #     traj[n] = x0 + v3 * i - 0.5 * gr0 * i ** 2

        else:
            # traj[n] =x0 + v1 * i - 0.5 * gr0 * i ** 2

            traj[n] = x0 - traj[int(len(t) / 2) - 1] + v1 * i - 0.5 * gr0 * i ** 2
    return traj


m_p = 1.44316072 * 10 ** -25
hbar_p = 1.0545718 * 10 ** -34
gr_p = 9.81
sig_p = 10 ** -3
T_p = 150 * 10 ** -3
Vk_p = 0.1
x0_p = 0
V0_p = 0
xlim_p = 0.5

m_s = 1
T_s = 1
L_s = 1

########## Applying Scalings
N_atom = 1
m = N_atom * m_p / m_s
hbar = hbar_p * T_s / (m_s * L_s ** 2)
gr = gr_p * T_s ** 2 / L_s
T = T_p / T_s
sig = sig_p / L_s
vkick = Vk_p / L_s * T_s
x0 = x0_p / L_s
V0 = V0_p / L_s * T_s
a0 = 1j * hbar / (4 * sig ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)

keff = m * vkick / hbar

N = 2 ** 10
dx = xlim_p / N
x = np.asarray([xlim_p - i * dx for i in range(N)])

Ns = 10000
dt = 2 * T_p / Ns
tl = np.asarray([i * dt for i in range(Ns)])

######## These Trajectories are Fucked
# traj1 = trajectory(tl, xlim_p, Vk_p, 0, 0, gr_p)
# traj2 = trajectory(tl, xlim_p, 0, Vk_p, 0, gr_p)
#
# p1 = keff*traj1[0]
# p2 = keff*traj1[int(Ns / 2)-1] - keff*traj2[int(Ns / 2)-1]
# p3 = keff*traj1[-1]
#
# dp = p1-2*p2+p3
#
# plt.plot(tl, traj1)
# plt.plot(tl, traj2)
# plt.show()


########### Heller
# args = m, hbar, gr
#
# init = [x0, V0, a0, g0]
# init2 = [x0, vkick, a0, g0]
# init_array = np.array([init, init2])
#
# Sole_hel = mul.HellerInterference(Ns, dt, init_array, derivsdt, args, x)
# arr = Sole_hel.onePacketArr(vkick, init)
# arr2 = Sole_hel.onePacketArr(-vkick, init)

# psi1 = Sole_hel.hellerPacketVal(arr[-1])
# psi2 = Sole_hel.hellerPacketVal(arr2[-1])
# plt.plot(x, psi1)
# plt.plot(x, psi2)
# plt.show()

# plotBasic(tl, arr)
# plotBasic(tl,arr2)
# plotBasic2(tl,arr,arr2)

#
# traj1 = arr[:, 0]
# traj2 = arr2[:, 0]

##### Path Plot
# plt.plot(tl, traj1)
# plt.plot(tl, traj2)
# plt.axvline(tl[int(Ns / 2)], color="k", linestyle=":")
# plt.show()

###### Phase plot
# traj1 = arr[:, 3]
# traj2 = arr2[:, 3]
#
# plt.plot(tl,(traj1-traj2)/hbar)
# plt.show()


# traj1 = xlim_p+vkick*tl-0.5*gr*tl**2
# p1 = keff * (traj1[0] + traj2[0]) / 2
# p2 = keff * (traj1[int(Ns / 2)] + traj2[int(Ns / 2)]) / 2
# p3 = keff * (traj1[-1] + traj2[-1]) / 2
#
# dp = p1 - 2 * p2 + p3
#
# print(dp / (keff * T ** 2))


################### MZ Test
args = m, hbar, gr
init = [x0, V0, a0, g0]
init2 = [x0, V0, a0, g0]
init_array = np.array([init, init2])
Solve_hel = mul.HellerInterference(Ns, dt, init_array, derivsdt, args, x)

# phi_arr = np.asarray([0, 10**7, 0])/hbar
phi_arr = np.asarray([0, 0, 0]) / hbar
d_phi = phi_arr[0] - 2 * phi_arr[1] + phi_arr[2]
omeg_eff = 0

tl, dat1, dat2 = Solve_hel.machZender(vkick, omeg_eff, phi_arr)

# plotBasic(tl, dat1)
# plotBasic(tl, dat2)
plotBasic2(tl, dat1, dat2)

psi1 = Solve_hel.hellerPacketVal(dat1[-1])
psi1s = psi1*np.conjugate(psi1)
psi2 = Solve_hel.hellerPacketVal(dat2[-1])
psi2s = psi1*np.conjugate(psi2)

# plt.plot(x,psi1s)
# plt.plot(x,psi2)
# plt.show()

p1 = dat1[-1, 3]
p2 = dat2[-1, 3]

print((gr - ((p2 - p1) + hbar * d_phi) / (keff * T ** 2)))
print(gr - ((p2 - p1) / (keff * T ** 2)))

########
# const = keff * gr * T ** 2
# # print(np.pi / const)
# #
# dp = np.linspace(-10, 10, 1000)
# signal = 0.5 * (1 - np.cos(const - dp))
# plt.plot(dp, signal)
# plt.scatter(d_phi, 0.5 * (1 - np.cos((p2 - p1)-hbar*d_phi)))
# plt.show()
