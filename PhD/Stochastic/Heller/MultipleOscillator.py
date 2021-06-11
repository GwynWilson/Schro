from PhD.Stochastic.Heller import MultiplePackets as mul
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps


def derivs(t, current, args, eta, dt):
    m, hbar, w = args
    x = current[1]
    v = -w ** 2 * current[0]
    a = (-2 * current[2] ** 2 / m) - (m * w ** 2) / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * current[0] ** 2 / 2
    return x, v, a, g


N = 2 ** 10
dx = 0.07
x_length = N * dx
x_length_half = int(x_length / 2)
x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx
x = x - x_length_half

m = 1
hbar = 1
w = 0.1
args = (m, hbar, w)

d = 2
k_initial = 2

vKick = k_initial * hbar / m
p0 = 0
a0 = 1j * hbar / (4 * d ** 2)
# a0 = 1j * m * w / 2
g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)

t = 0
dt = 0.01
step = 1
Ns = 2000
Ntot = int(Ns * step)
N_half = int(Ntot / 2)

x0 = x01 = 0
init = [x0, p0 / m, a0, g0]
init2 = [x01, p0 / m, a0, g0]
init_array = np.array([init, init2])

###### One
# init_array = np.array([init])
# helpack = mul.HellerInterference(N, dt, init_array, derivs, args, x)
# helpack.runAll()
# mods = helpack.modSquare()
#
# normlist = []
# for i in mods:
#     temp_norm = simps(i,x)
#     normlist.append(temp_norm)
#
# plt.plot(normlist)
# plt.show()

# plt.plot(x,mods[0])
# plt.plot(x,mods[-1])
# plt.show()

########## Heller interference
# helpack = mul.HellerInterference(Ntot, dt, init_array, derivs, args, x)
# helpack.interferometry2(vKick,psii=True)
# mods = helpack.modSquare()
#
# plt.figure()
# # plt.title("Wave packet interference for time T")
# plt.plot(x, mods[0], label="t=0")
# plt.plot(x, mods[int(Ntot/4)], label="t=T/4")
# plt.plot(x, mods[int(2*Ntot/4)], label="t=T/2")
# plt.plot(x, mods[Ntot], label="t=T")
# plt.xlabel("x")
# plt.ylabel(r"$|\psi(x)|^2$")
# plt.xlim(min(x), max(x))
# plt.ylim(0, max(mods[0]))
# plt.legend()
# # plt.savefig("Heller_split_wave.png")
# plt.show()

########### Schro Interference
# V = 0.5 * m * w ** 2 * x ** 2
#
# schpack = mul.SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
# schpack.interferometry(vKick,psii=True)
# mods = schpack.modSquare()
# plt.figure()
# plt.title("Wave packet interference for time T")
# plt.plot(x, mods[0], label="t=0")
# # plt.plot(x, mods[int(N/4)], label="t=T/4")
# plt.plot(x, mods[N_half], label="t=T/2")
# plt.plot(x, mods[int(3*N/4)], label="t=3T/4")
# plt.plot(x, mods[-1], label="t=T")
# plt.xlabel("x")
# plt.ylabel(r"$|\psi(x)|^2$")
# plt.xlim(min(x), max(x))
# plt.ylim(0, max(mods[0]))
# plt.legend()
# # plt.savefig("Heller_split_wave.png")
# plt.show()


##### Full interferance
V = 0.5 * m * w ** 2 * x ** 2
sch_tupac = mul.SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
sch_tupac.interferometry(vKick)
sch_mods = sch_tupac.modSquare()

hel_tupac = mul.HellerInterference(Ntot, dt, init_array, derivs, args, x)
hel_tupac.interferometry2(vKick)
hel_mods = hel_tupac.modSquare()

comp = mul.Comparison(Ntot, dt, x, sch_tupac.psi_comb, hel_tupac.psi_comb)
comp.animate(potential=V)
comp.bothNorm()
# comp.plotHalf()
comp.fullOverlap()


