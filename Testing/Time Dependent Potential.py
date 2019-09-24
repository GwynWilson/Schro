import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def xpos(t, xi, omega, a, x0):
    w = omega
    return (xi + (a * w ** 2)) * np.cos(w * t) + (0.5 * a * (t ** 2)) + x0 - (a * w ** 2)


def wave_init(x, sig, x0):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - x0) ** 2) / (4 * sig ** 2))


def harmonic_potential_t(x, t, args=(1, 1, 1, 1)):
    """

    :param x:
    :param t:
    :param args: [0]=mass,[1]=omega,[2]=x0,[3]=a
    :return:
    """
    return 0.5 * args[0] * args[1] ** 2 * (x - args[2] - 0.5 * args[3] * t ** 2) ** 2


def potential(x, t=0):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        temp[n] = 0.01 * t
    return temp


# Defining x axis
N = 2 ** 12
dx = 0.05
x_length = N * dx

x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx

x0 = int(0.1 * x_length)

hbar = 1
m = 1
omega = 1
sigma = np.sqrt(hbar / (2 * m * omega))
a = 1
args = (m, omega, x0, a)
K = 0.5 * m * omega ** 2

# Defining Psi and V
k_initial = 10
xi = -10
A = x0 + xi
psi_x = wave_init(x, sigma, A)
V_x = harmonic_potential_t(x, 0, args)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 5
Ns = 350
print(Ns*step*dt)
sch = Schrodinger(x, psi_x, harmonic_potential_t, k, hbar=hbar, m=m, t=t, args=args)

# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.ylim(0, max(np.real(psi_x)))
# plt.show()
#
# t_list = []
#
# expec_x = []
#
# for i in range(Ns):
#     if i != 0:
#         sch.evolve_t(step, dt)
#     t_list.append(sch.t)
#     expec_x.append(sch.expectation_x())
#
# x_list = [xpos(j, xi, omega, a, x0) for j in t_list]
# diff = [abs(x_list[k] - expec_x[k]) for k in range(len(x_list))]
#
# plt.plot(t_list, expec_x, label='Calculated x')
# plt.plot(t_list, x_list, linestyle='--', label='Expected x')
# plt.title('Expectation value of x over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('Expec_X.png')
# plt.show()
#
# plt.plot(t_list, diff)
# plt.title('Difference between calculated x and expected x')
# plt.xlabel('Time')
# plt.ylabel(r'$x - <x>$')
# plt.savefig('Expec_X_diff.png')
# plt.show()

a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_x)))),
            lim2=((ks[0], ks[N - 1]), (0, 30)))
a.make_fig()
