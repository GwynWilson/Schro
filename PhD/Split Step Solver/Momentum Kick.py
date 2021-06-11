import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import animation

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * (x-x0))


def gauss_init_pi(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(
        1j * 2 * np.pi * k0 * x)


def x_pos(t, x0, kini, hbar=1, m=1):
    return x0 + t * kini * (hbar / m)


def width(t, hbar=1, m=1, sigma=1):
    return sigma * np.sqrt(1 + (t ** 2) * (hbar / (2 * m * (sigma ** 2)) ** 2))


def func(x, a, b):
    return a * x + b


# Defining x axis
N = 2 ** 11
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx
x0 = int(0.25 * x_length)

d = 1

# Defining Psi and V
k_initial = 4
psi_x = gauss_init(x, k_initial, x0, d=d)
V_x = np.zeros(N)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 10
Ns = 100
print("Final time", dt * step * Ns)

hbar = 1
m = 1

sch = Schrodinger(x, psi_x, V_x, k, hbar=hbar, m=m)

print("Numerical Energy :", sch.energy())
print("Theoretical Energy", (hbar ** 2 * k_initial ** 2) / (2 * m))

print("Difference :", sch.energy() - (hbar ** 2 * k_initial ** 2) / (2 * m))

psi_init2 = gauss_init(x, k_initial, x0=x0, d=d)
psis2 = np.real(psi_init2 * np.conj(psi_init2))

# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.ylim(0, max(np.real(psi_x)))
# plt.show()

# a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_x)))),
#             lim2=((ks[0], ks[N-1]), (0, 1)))
# a.make_fig()


t_list = []
norm_x = []
expec_x = []
expec_xs = []
expec_k = []

for i in range(int(Ns/2)):
    if i != 0:
        sch.evolve_t(step, dt)
    t_list.append(sch.t)
    norm_x.append(sch.norm_x() - 1)
    expec_x.append(sch.expectation_x())
    expec_xs.append(np.sqrt(sch.expectation_x_square() - sch.expectation_x() ** 2))
    expec_k.append(sch.expectation_k())

sch.momentum_kick(-2*k_initial)

for i in range(int(Ns/2)):
    sch.evolve_t(step, dt)
    t_list.append(sch.t)
    norm_x.append(sch.norm_x() - 1)
    expec_x.append(sch.expectation_x())
    expec_xs.append(np.sqrt(sch.expectation_x_square() - sch.expectation_x() ** 2))
    expec_k.append(sch.expectation_k())



x_pos_list = [x_pos(j, x0, k_initial, hbar=hbar, m=m) for j in t_list]
# xdiff = [np.abs(expec_x[n] - x_pos_list[n]) for n in range(len(expec_x))]
#
# popt1, pcov = curve_fit(func, t_list, x_pos_list)
# print("Expected x :", popt1)
#
# popt2, pcov = curve_fit(func, t_list, expec_x)
# print("Calculated x :", popt2)

# plt.plot(t_list, norm_x, linestyle='none', marker='x')
# plt.title('Normalistaion of wavefunction over time')
# plt.xlabel('Time')
# plt.ylabel('Normalisation-1')
# plt.savefig('Normalisation.png')
# plt.show()

plt.plot(t_list, expec_x, label='Calculated x')
plt.plot(t_list, x_pos_list, linestyle='--', label='Expected x')
plt.title('Expectation value of x over time')
plt.xlabel('Time')
plt.ylabel(r'$<x>$')
plt.legend(loc='best', fancybox=True)
plt.savefig('Expec_X_lin.png')
plt.show()

# plt.plot(t_list, xdiff, linestyle='none', marker='o', markersize=1, label='Difference in x')
# plt.title('Difference between calculated x and expected x')
# plt.xlabel('Time')
# plt.ylabel(r'$x - <x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('Expec_X_diff_lin.png')
# plt.show()

widthx = [width(j, sigma=np.sqrt(d), hbar=hbar, m=m) for j in t_list]
widthdiff = [abs(widthx[n] - expec_xs[n]) for n in range(len(widthx))]

plt.plot(t_list, expec_xs, label='Calculated width')
plt.plot(t_list, widthx, linestyle='--', label='Expected width')
plt.legend(loc='best', fancybox=True)
plt.title('Width of distribution over time')
plt.xlabel('Time')
plt.ylabel(r'$<\Delta x>$')
plt.savefig('delta_x_lin.png')
plt.show()

# plt.subplots_adjust(left=0.16)
# plt.plot(t_list, widthdiff, linestyle='none', marker='o', markersize=1, label='Difference in x')
# plt.title('Difference between calculated width and expected width')
# plt.xlabel('Time')
# plt.ylabel(r'$\Delta x - <\Delta x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('delta_X_diff_lin.png')
# plt.show()

kt = [k_initial for i in t_list]
k_diff = [expec_k[i] - kt[i] for i in range(len(t_list))]
plt.figure()
plt.plot(t_list, expec_k, label="Expectation K")
plt.plot(t_list, [k_initial for i in t_list], label="Theoretical k", linestyle="--")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(t_list, k_diff, linestyle='none', marker='o', markersize=1)
# plt.show()
