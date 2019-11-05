from Schrodinger_Solver import Schrodinger
from Animate import Animate
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


def gauss_barrier(x, A, x0, w):
    return A * (np.exp(-(x - x0) ** 2 / w ** 2))


def doubleGauss(x, A, omeg, x1, x2):
    # return gauss_barrier(x, A, x1, omeg) + gauss_barrier(x, A, x2, omeg)
    return A * (np.exp(-(x - x1) ** 2 / omeg ** 2)) + A * (np.exp(-(x - x2) ** 2 / omeg ** 2))


def barrier(x, A, x1, x2):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        elif v > x2:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


def doubleBarrier(x, A, x1, wid, sep):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        # Left Barrier
        if (v > x1 - (sep / 2) - wid and v < x1 - (sep / 2)):
            temp[n] = A
        elif (v > x1 + (sep / 2) and v < (x1 + sep / 2 + wid)):
            temp[n] = A
        else:
            temp[n] = 0
        # print(sep)
        # if v < x1 - (sep / 2):
        #     print("Here")
    return temp


def transmissionSpectrum(min, max, points, V):
    sch = Schrodinger(x, Psi_x, V, hbar=hbar, m=m, t=0)
    energies = np.linspace(min, max, points) * bar_amp
    transmissions = []
    for e in energies:
        imp = sch.impedence(e)
        transmissions.append(imp)

    return energies, transmissions


def ashbyTunnling(E, V1, V2, a, b):
    K = np.sqrt(2 * m * E / hbar ** 2)
    k1 = np.sqrt(2 * m * (V1 - E) / hbar ** 2 + 0j)
    k2 = np.sqrt(2 * m * (V2 - E) / hbar ** 2 + 0j)
    delta = a - b

    B = np.exp(-2j * K * b) * ((K ** 2 / k2 + k2) * np.exp(2j * K * a) * np.cosh(k1 * delta) * np.sinh(k2 * delta)
                               + (K ** 2 / k1 + k1) * np.exp(-2j * K * a) * np.sinh(k1 * delta) * np.cosh(k2 * delta)
                               + ((K ** 3 / (k1 * k2) - k1 * k2 / K) * np.sin(2 * K * a) + 1j * K * (
                    k1 / k2 - k2 / k1) * np.cos(2 * K * a)) * np.sinh(k1 * delta) * np.sinh(k2 * delta))

    A = (np.exp(-2j * K * a) * ((K ** 2 / k2 - k2) * np.cosh(k1 * delta) * np.sinh(k2 * delta)
                                + (K ** 2 / k1 - k1) * np.sinh(k1 * delta) * np.cosh(k2 * delta) - 2j * K * np.cosh(
                k2 * delta) * np.cosh(k1 * delta))
         + ((K ** 3 / (k1 * k2) + k1 * k2 / K) * np.sin(2 * K * a) - 1j * K * (k1 / k2 + k2 / k1) * np.cos(
                2 * K * a)) * np.sinh(k2 * delta) * np.sinh(k1 * delta))

    if A == 0:
        A = 10 ** -99

    return 1 - abs(B / A) ** 2


Psi_x = gauss_init(x, k0, x0=x0, d=sig)

V_x = doubleBarrier(x, bar_amp, x1, wid, sep)
# V_x = np.zeros(len(x))

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0)

######## Initial Plot
# plt.plot(x / 10 ** -6, sch.psi_squared)
# plt.plot(x / 10 ** -6, V_x / scale)
# plt.show()


######## Animate
# a = Animate(sch, V_x, step, dt, lim1=((x[0], x[-1]), (0, max(np.real(sch.psi_squared)))),
#             lim2=((sch.k[0], sch.k[-1]), (0, max(np.real(sch.psi_k)))))
# a.make_fig()

#############Ashby Tunneling Test
a = sep / 2
b = sep / 2 + wid
min = 0
max = 1
num = 1000

energies = np.linspace(min, max, num) * bar_amp
transmissions = []
for e in energies:
    transmissions.append(ashbyTunnling(e, bar_amp, bar_amp, a, b))
energies_imp, trans_imp = transmissionSpectrum(min, max, num, V_x)

plt.plot(energies / bar_amp, transmissions, label="Analytic")
plt.plot(energies_imp / bar_amp, trans_imp, label="Numerical")
plt.yscale("log")
plt.xlabel("Energy/v0")
plt.ylabel("Transmission Probability")
plt.legend(loc="best", fancybox="tre")
plt.savefig("Asby_Resonance")
plt.show()

########### Square Barrier Spectrum
# energies, transmissions = transmissionSpectrum(0, 1, 1000, V_x)
#
# plt.plot(energies / bar_amp, transmissions)
# plt.xlabel("Energy/v0")
# plt.ylabel("Transmission Probability")
# plt.savefig("Impedence_Resonance")
# plt.show()
#
# plt.plot(energies / bar_amp, transmissions)
# plt.yscale("log")
# plt.xlabel("Energy/v0")
# plt.ylabel("Transmission Probability")
# plt.savefig("Impedence_Resonance_log")
# plt.show()


######## Gauss barier spectrum
# v_g = doubleGauss(x, bar_amp, omeg, -separation/2, separation/2)
# energies, transmissions = transmissionSpectrum(0, 1, 1000, v_g[::])
#
# plt.plot(energies / bar_amp, transmissions)
# plt.xlabel("Energy/v0")
# plt.ylabel("Transmission Probability")
# plt.savefig("Impedence_Resonance_Gauss")
# plt.show()
#
# plt.plot(energies / bar_amp, transmissions)
# plt.yscale("log")
# plt.xlabel("Energy/v0")
# plt.ylabel("Transmission Probability")
# plt.savefig("Impedence_Resonance_log_Gauss")
# plt.show()
