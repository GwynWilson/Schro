import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def wave_init(x, sig, x0):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - x0) ** 2) / (4 * sig ** 2))


def wave_init_2(x, sig, x0, k0=0):
    return 1 / ((2 * np.pi * (sig ** 2)) ** (1 / 4)) * np.exp(-((x - x0) ** 2) / (4 * sig ** 2)) * np.exp(1j * k0 * x)


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * x)

def gauss_interf(x,k0,x1,d=1):
    return gauss_init(x,k0,x1) + gauss_init(x,-k0,-x1)


def gaussbox(x, w, L, x0=0, A=1, a=0):
    temp = A * (np.exp(-(x - x0) ** 2 / w ** 2) + np.exp(-(x - (x0 + L)) ** 2 / w ** 2)) + m * a * x
    return temp


def run(acc):
    t = 0
    psi_x = gauss_init(x, k_initial, x0=0, d=d)
    V_x = gaussbox(x, w, L, x0=xb, A=Amp, a=acc)
    sch = Schrodinger(x, psi_x, V_x, k, hbar=hbar, m=m, t=t)

    t_list = []
    x_list = []

    for i in range(Ns):
        sch.evolve_t(N_steps=step, dt=dt)
        t_list.append(sch.t)
        x_list.append(sch.expectation_x())

    dat = list(zip(t_list, x_list))
    # np.savetxt("GBox_{}.txt".format(a), dat)
    return dat


# Defining x axis
N = 2 ** 12
dx = 0.05
x_length = N * dx
x0 = int(0.5 * x_length)
x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx - x0

# x0 = 0
# Defining constants
hbar = 1
m = 1
omega = 1
sigma = np.sqrt(hbar / (2 * m * omega))
d = 4
a = 10

# Defining Psi and V
k_initial = 10
Amp = 1000
w = 1
L = int(0.4 * x_length)
xb = -L / 2

# K=10,A=10**1.7

psi_x = gauss_init(x, k_initial, x0=0, d=d)
psi_x = gauss_interf(x,k_initial,xb,d=d)
V_x = gaussbox(x, w, L, x0=xb, A=Amp, a=a)
V_x= np.zeros(len(psi_x))

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.001
step = 50
Ns = 400

sch = Schrodinger(x, psi_x, V_x, k, hbar=hbar, m=m, t=t)

# print("Numerical Energy :", sch.energy())
# print("Theoretical Energy", (hbar ** 2 * k_initial ** 2) / (2 * m))
#
# print("Difference :", sch.energy() - (hbar ** 2 * k_initial ** 2) / (2 * m))

# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.ylim(0, max(np.real(psi_x)))
# plt.show()

print("here")
# a = Animate(sch, V_x, step, dt, lim1=((-x_length / 4, x_length / 4), (-0.1, 0.5)),
#             lim2=((ks[0], ks[N - 1]), (0, 1)), title="GB_a=10", frames=Ns)
a = Animate(sch, V_x, step, dt, lim1=((-x_length / 4, x_length / 4), (-0.1, 0.5)),
            lim2=((ks[0], ks[N - 1]), (0, 1)), frames=Ns)
a.make_fig()

# a_list = np.arange(1, 10, 2)
# a_list=[10,0.1]
# for i in a_list:
#     print(i)
#     dat = run(i)
#     np.savetxt("GBox_{}.txt".format(i), dat)


# v0 = V_x = gaussbox(x, w, L, x0=xb, A=1000, a=0)
# v1 = V_x = gaussbox(x, w, L, x0=xb, A=1000, a=1)
#
# plt.plot(x, v0)
# plt.title("Gaussian Barrier a=0")
# plt.xlabel("x")
# plt.ylabel("V(x)")
# plt.savefig("GB_a=0")
# plt.show()
#
# plt.plot(x, v1)
# plt.title("Gaussian Barrier a=1")
# plt.xlabel("x")
# plt.ylabel("V(x)")
# plt.savefig("GB_a=1")
# plt.show()
