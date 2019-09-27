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


def harmonic_potential_t(x, t, args=(1, 1, 1, 1, 1)):
    """
    :param x:
    :param t:
    :param args: [0]=mass,[1]=omega,[2]=a,[3]=x0, [4] = c
    :return:
    """
    return 0.5 * args[0] * (args[1] ** 2) * (x - args[3]) ** 2


def harmonic_potential(x, t, args=(1, 1, 1, 1, 0)):
    """
    :param x:
    :param t:
    :param args: [0]=mass,[1]=omega,[2]=a, [3]=x0, [4] = c
    :return:
    """
    m, w, a, x0, c = args
    return 0.5 * m * (w ** 2) * ((x - x0) ** 2) + m * a * x + c * ((x - x0) ** 3)
    # return 0.5 * m * (w ** 2) * ((x - x0) ** 2) + m * a * x + 0.5 * m * (a ** 2) * (t ** 2)


def run(Ns, step, dt, acc, c):
    args = (m, omega, acc, x0, c)
    v = harmonic_potential(x, t, args=args)
    sch = Schrodinger(x, psi_x, v, k, hbar=hbar, m=m, t=0, args=args)
    for i in range(Ns):
        sch.evolve_t(N_steps=step, dt=dt)
    return sch.expectation_x()


def accl(acclist, c):
    x_list = []
    for j in acclist:
        print(j)
        x_list.append(run(Ns, step, dt, j, c))
    return list(zip(acclist, x_list))


# Defining x axis
N = 2 ** 12
dx = 0.05
x_length = N * dx
x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx - x_length / 2
x0 = 0

# Defining constants
hbar = 1
m = 1
omega = 1
sigma = np.sqrt(hbar / (2 * m * omega))
a = 1
c = -0.01
args = (m, omega, a, x0, c)
args2 = (m, omega, a, x0, 0)

# Defining Psi and V
k_initial = 10
xi = -20
A = x0 + xi
psi_x = wave_init(x, sigma, A)
V_x = harmonic_potential(x, 0, args)
V_t = harmonic_potential_t(x, 0, args)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.001
step = 50
Ns = 63
print(Ns * step * dt)

sch = Schrodinger(x, psi_x, harmonic_potential, k, hbar=hbar, m=m, t=t, args=args)

# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.plot(x, harmonic_potential(x, t, args=args2), linestyle="--")
# plt.plot(x,V_t)
# plt.ylim(0, max(np.real(psi_x)))
# plt.show()

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
# plt.plot(t_list, expec_x, label='Calculated x')
# plt.title('Expectation value of x over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('Expec_X_trans.png')
# plt.show()
#
# shift = [0 for k in t_list]
# xpos_list = [xpos(j, xi, omega, a, x0) for j in t_list]
#
# for n, v in enumerate(expec_x):
#     shift[n] = expec_x[n] + 0.5 * a * (t_list[n] ** 2)
#
# plt.plot(t_list, shift, label='Shifted x')
# plt.plot(t_list, xpos_list, linestyle='--', label='Classical')
# plt.title('Expectation value of x over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<x>$')
# plt.legend(loc='best', fancybox=True)
# plt.savefig('Shifted_x.png')
# plt.show()
#
# diff = [abs(xpos_list[k] - shift[k]) for k in range(len(t_list))]
# plt.plot(t_list, diff)
# plt.title('Difference between calculated x and expected x')
# plt.xlabel('Time')
# plt.ylabel(r'$x - <x>$')
# plt.savefig('Shifted_x_diff.png')
# plt.show()

# a = Animate(sch, V_x, step, dt, lim1=((-x_length / 2, x_length / 2), (0, max(np.real(psi_x)))),
#             lim2=((ks[0], ks[N - 1]), (0, 2)))
#
# a.make_fig()

c_list = [-0.01, 0.01]
a_list = np.arange(0, 10, 0.5)

for c in c_list:
    print("c", c)
    dat = accl(a_list, c)
    np.savetxt("HO_{}.txt".format(c), dat)
