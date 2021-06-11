from PhD.Stochastic.Heller.MultiplePackets import SchroInterference, HellerInterference, Comparison
import matplotlib.pyplot as plt
import numpy as np


def derivs(t, current, args, eta, dt):
    m, hbar = args
    x = current[1]
    v = 0
    a = (-2 * current[2] ** 2 / m)
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2
    return x, v, a, g


def psi_t(t, x, x0, v0, a0, g0, m, hbar):
    t = np.asarray(t)
    xt = x0 + v0 * t
    vt = np.asarray([v0 for k in range(len(t))])
    at = a0 / ((2 / m) * a0 * t + 1)
    gt = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2) + 1j * hbar / 2 * np.log((2 / m) * a0 * t + 1)
    psi_comb = np.zeros((len(t), len(x)), dtype=complex)
    for i in range(len(t)):
        psi_temp = np.exp(
            (1j / hbar) * at[i] * (x - xt[i]) ** 2 + (1j / hbar) * m * vt[i] * (x - xt[i]) + (
                    1j / hbar) * gt[i])
        psi_comb[i] = psi_temp
    return psi_comb


###################### Variables
N = 2 ** 10
dx = 0.1
x_length = N * dx
x = np.asarray([i * dx for i in range(N)])
x0 = int(0.25 * x_length)

m = 1
hbar = 1
args = (m, hbar)

t = 0
dt = 0.01
step = 1
Ns = 1000
Ntot = int(Ns * step)
N_half = int(Ntot / 2)
V = np.zeros(N)

################## Standing Packet
# d = 2
# v_init = 0
#
# x0 = int(3 / 8 * x_length)
# x01 = int(5 / 8 * x_length)
# v0 = v_init
# a0 = 1j * hbar / (4 * d ** 2)
# g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)
# init1 = [x0, v0, a0, g0]
# init2 = [x01, v0, a0, g0]
# init_array = np.array([init1, init2])
#
# Sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
# Sch_tupac.runAll(step)
# Sch_psi = Sch_tupac.psi_comb
#
# Hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
# Hel_tupac.runAll()
# Hel_psi = Hel_tupac.psi_comb
#
# Comp = Comparison(Ntot, dt, x, Sch_psi, Hel_psi)
# Comp.plotInit()
# Comp.bothNorm()
# Comp.fullOverlap()

#################### Single moving packet
d = 1
v_init = 2

x0 = int(0.25 * x_length)
v0 = v_init
a0 = 1j * hbar / (4 * d ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)
init1 = [x0, v0, a0, g0]
init_array = np.array([init1])

t = [i * dt for i in range(Ntot)]
analytic = psi_t(t, x, x0, v0, a0, g0, m, hbar)

Sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
Sch_tupac.runAll(step)
Sch_psi = Sch_tupac.psi_comb

Hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
Hel_tupac.runAll()
Hel_psi = Hel_tupac.psi_comb

########### Psi comparison Initial
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(x, np.real(Sch_psi[0]), label="sch")
# axs[0].plot(x, np.real(Hel_psi[0]), label="hel")
# axs[1].plot(x, np.imag(Sch_psi[0]))
# axs[1].plot(x, np.imag(Hel_psi[0]))
#
# axs[0].plot(x, np.real(analytic[0]), label="Theory", linestyle="--")
# axs[1].plot(x, np.imag(analytic[0]), linestyle="--")
#
# axs[0].legend()
# plt.show()

############### Psi Comparison Final
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(x, np.real(Sch_psi[-1]), label="sch")
# axs[0].plot(x, np.real(Hel_psi[-1]), label="hel")
# axs[1].plot(x, np.imag(Sch_psi[-1]))
# axs[1].plot(x, np.imag(Hel_psi[-1]))
#
# axs[0].plot(x, np.real(analytic[-1]), label="Theory", linestyle="--")
# axs[1].plot(x, np.imag(analytic[-1]), linestyle="--")
#
# axs[0].legend()
# plt.show()

####### Comparison
Comp = Comparison(Ntot, dt, x, Sch_psi, Hel_psi)
mod_analytic = np.conjugate(analytic) * analytic
Comp.plotInit()
Comp.fullOverlap()
