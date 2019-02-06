import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import animation

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * x)


def x_pos(t, x0, kini, hbar=1, m=1):
    return x0 + t * kini * (hbar / m)


def width(t, hbar=1, m=1, sigma=1):
    return sigma * np.sqrt(1 + (t ** 2) * (hbar / 2 * m * (sigma ** 2)) ** 2)


def func(x, a, b):
    return a * x + b


# Defining x axis
N = 2 ** 11
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x0 = int(0.25 * x_length)

d = 1

# Defining Psi and V
k_initial = 10
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

sch = Schrodinger(x, psi_x, V_x, k)

"""
plt.plot(x, sch.mod_square_x(True))
plt.plot(x, V_x)
plt.ylim(0, max(np.real(psi_x)))
plt.show()

a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_x)))),
            lim2=((ks[0], ks[N-1]), (0, 30)))
a.make_fig()
"""

t_list = []
norm_x = []
expec_x = []
expec_xs = []

for i in range(100):
    if i != 0:
        sch.evolve_t(step, dt)
    t_list.append(sch.t)
    norm_x.append(sch.norm_x() - 1)
    expec_x.append(sch.expectation_x())
    expec_xs.append(np.sqrt(sch.expectation_x_square() - expec_x[i] ** 2))

x_pos_list = [x_pos(j, x0, k_initial) for j in t_list]
xdiff = [np.abs(expec_x[n] - x_pos_list[n]) for n in range(len(expec_x))]

popt1, pcov = curve_fit(func, t_list, x_pos_list)
print("Expected x :", popt1)

popt2, pcov = curve_fit(func, t_list, expec_x)
print("Calculated x :", popt2)

plt.plot(t_list, norm_x, linestyle='none', marker='x')
plt.title('Normalistaion of wavefunction over time')
plt.xlabel('Time')
plt.ylabel('Normalisation-1')
plt.savefig('Normalisation.png')
plt.show()



plt.plot(t_list, expec_x, label='Calculated x')
plt.plot(t_list, x_pos_list, linestyle='--', label='Expected x')
plt.title('Expectation value of x over time')
plt.xlabel('Time')
plt.ylabel(r'$<x>$')
plt.legend(loc='best', fancybox=True)
plt.savefig('Expec_X.png')
plt.show()

plt.plot(t_list, xdiff, linestyle='none', marker='o', markersize=1, label='Difference in x')
plt.title('Difference between calculated x and expected x')
plt.xlabel('Time')
plt.ylabel(r'$x - <x>$')
plt.legend(loc='best', fancybox=True)
plt.savefig('Expec_X_diff.png')
plt.show()

widthx = [width(j, sigma=np.sqrt(d)) for j in t_list]
widthdiff = [abs(widthx[n] - expec_xs[n]) for n in range(len(widthx))]

plt.plot(t_list, expec_xs, label='Calculated width')
plt.plot(t_list, widthx, linestyle='--', label='Expected width')
plt.legend(loc='best', fancybox=True)
plt.title('Width of distribution over time')
plt.xlabel('Time')
plt.ylabel(r'$<\Delta x>$')
plt.savefig('delta_x.png')
plt.show()

plt.subplots_adjust(left=0.16)
plt.plot(t_list, widthdiff, linestyle='none', marker='o', markersize=1, label='Difference in x')
plt.title('Difference between calculated width and expected width')
plt.xlabel('Time')
plt.ylabel(r'$\Delta x - <\Delta x>$')
plt.legend(loc='best', fancybox=True)
plt.savefig('delta_X_diff.png')
plt.show()
