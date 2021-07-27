from PhD.Stochastic.Heller import MultiplePackets as mul
from PhD.Stochastic.Heller import Heller as hel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps


def hellerPacket(xt, vt, at, gt):
    return np.exp(
        (1j / hbar) * at * (x - xt) ** 2 + (1j / hbar) * m * vt * (x - xt) + (
                1j / hbar) * gt)


def derivs(t, current, args, eta, dt):
    m, hbar = args
    x = current[1] * dt
    v = 0
    a = (-2 * current[2] ** 2 / m) * dt
    g = (1j * hbar * current[2] / m + m * current[1] ** 2 / 2) * dt
    return x, v, a, g


def derivsdt(t, current, args, eta, dt):
    m, hbar, w, sig = args
    x = current[1] * dt
    v = -w ** 2 * current[0] * dt
    a = (-2 * current[2] ** 2 / m) * dt - (m * w ** 2) * dt / 2
    g = 1j * hbar * current[2] * dt / m + m * current[1] ** 2 * dt / 2 - m * w ** 2 * current[0] ** 2 * dt / 2
    return x, v, a, g


def derivsStocdt(t, current, args, eta, dt):
    m, hbar, w, sig_n = args
    x = current[1] * dt
    v = -w ** 2 * (current[0] * dt - eta * sig_n)
    a = ((-2 * current[2] ** 2 / m) - (m * w ** 2) / 2) * dt
    g = (1j * hbar * current[2] / m + m * current[1] ** 2 / 2) * dt - 0.5 * m * w ** 2 * current[
        0] ** 2 * dt + m * w ** 2 * current[0] * eta * sig_n - 0.5 * m * w ** 2 * sig_n ** 2 * eta ** 2
    return x, v, a, g


def expected(t, args, init):
    t = np.asarray(t)
    m, hbar, w, sig_n = args
    x0, v0, a0, g0 = init
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t)
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))
    g_ex = g0 - hbar * w * t / 2 + 0.5 * m * (v_ex * x_ex - v0 * x0)
    return x_ex, v_ex, a_ex, g_ex


m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
args = (m, hbar)

lim = 1.5 * 10 ** - 5
N = 2 ** 12
dx = 2 * lim / N
# x = np.arange(-lim, lim, dx)
x = np.asarray([i * dx - lim for i in range(N)])

w = 100
sig = np.sqrt(hbar / (2 * m * w))  # Width of packet in 100Hz trap
print(sig)
x0 = 0
vel0 = 0
a0 = 1j * hbar / (4 * sig ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)
# g0 = -0.5j*hbar *np.log(1/(np.sqrt(2*np.pi*sig**2)))

init = [x0, vel0, a0, g0]
init2 = [x0, -vel0, a0, g0]
init_array = np.array([init, init2])

vkivk = 4 * 10 ** -3
T = 3 * 10 ** -3
Ns = 3000
dt = T / Ns

########### Iterferometry no potential
# Solve_hel = mul.HellerInterference(Ns, dt, init_array, derivs, args, x)
# Solve_hel.interferometry2(vkivk)
# mods = Solve_hel.modSquare()
# plt.plot(x/sig, mods[0]/max(mods[0]), label="t=0")
# plt.plot(x/sig, mods[int(Ns / 2)]/max(mods[0]), label="t=T/2")
# plt.plot(x/sig, mods[-1]/max(mods[0]), label="t=T")
# plt.xlabel(r"$x/\sigma$")
# plt.ylabel(r"$|\psi(x)|^2 / |\psi(0)|^2$")
# plt.legend()
# plt.show()

############ Heller
sigma_noise = 0.5 * sig
sigma_noise = 10**(-8)
print(sigma_noise/10**(-6))
args_stoc = (m, hbar, w, sigma_noise)

T = 4 * np.pi / w
Ns = 30000
dt = T / Ns

x0 = 20 * sig
v0 = 0
a0 = 1j * m * w / 2
g0 = (1j * hbar / 4) * np.log(2 * np.pi * sig ** 2)
init = [x0, v0, a0, g0]

solver = hel.Heller(Ns, dt, init, derivsStocdt)
tl, xl, vl, al, gl = solver.rk4dt(args_stoc)
tl = np.asarray(tl)
xl = np.asarray(xl)
vl = np.asarray(vl)
al = np.asarray(al)
gl = np.asarray(gl)
fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
axs[0, 0].plot(tl, xl / 10 ** -6)
axs[0, 1].plot(tl, vl / 10 ** -3)
axs[1, 0].plot(tl, np.imag(al) / np.imag(a0))
axs[1, 1].plot(tl, gl / hbar)
axs[0, 0].set_ylabel(r"$x(\mu m)$")
axs[0, 1].set_ylabel(r"$v(mms^{-1})$")
axs[1, 0].set_ylabel(r"$\alpha/\alpha_0$")
axs[1, 0].set_xlabel(r"$t(s)$")
axs[1, 1].set_ylabel(r"$\gamma/\hbar$")
axs[1, 1].set_xlabel(r"$t(s)$")
# plt.savefig("SHO_units")
plt.show()

# solver.averageRuns(100,args_stoc,dtver=True)
# solver.plotBasic(average=True,expected=expected)
