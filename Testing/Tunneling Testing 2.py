import numpy as np
from scipy.fftpack import fftfreq, fftshift
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate

import time


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (d ** 2)) * np.exp(1j * k0 * x)


def barrier(x, A, w, x0):
    temp = A * (np.exp(-(x - x0) ** 2 / w ** 2))
    return temp


def run(x2, time=False, a=1):
    V_x = barrier(x, a, x2, x1)
    psi_init = gauss_init(x, k_init, x0=x0, d=sig)

    sch = Schrodinger(x, psi_init, V_x, k, hbar=hbar, m=m, t=0, args=x1)

    dat = []
    t_list = []
    for i in range(0, Ns):
        sch.evolve_t(step, dt)
        dat.append(sch.barrier_transmition())
        t_list.append(sch.t)

    # plt.plot(sch.x, abs(sch.psi_x))
    # plt.plot(x, V_x)
    # plt.show()

    if time:
        return dat, t_list
    else:
        return dat


def multirun(var1, var2):
    t_list = 0
    dat = np.empty((len(var1), len(var2)))
    for i in range(len(var1)):
        for j in range(len(var2)):
            print(j)
            # dati.append(np.asarray(run(var2[j], time=False, a=var1[i])))
            datj, t_list = run(var2[j], time=True, a=var1[i])
            np.savetxt("{Amp}_{Wid}_gauss.csv".format(Amp=var1[i], Wid=var2[j]), datj, delimiter=",")
    np.savetxt("time_gauss.csv", t_list, delimiter=",")
    return 0


def run_w(wrange, Amp):
    Tunnel_Val = []
    for a in wrange:
        print("Amp: " + str(Amp) + "\nWidth: " + str(a))
        temp, t_list = run(a * dx, time=True, a=Amp)
        Tunnel_Val.append(max(temp))
        # Tunnel_Val.append(temp[-1])
    return Tunnel_Val


# Defining x axis
N = 2 ** 11
dx = 0.1
x_length = N * dx

x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx

hbar = 1
m = 1

# Barrier Definitions
A = 20
x1 = int(0.5 * N) * dx
W = 1
V_x = barrier(x, A, W, x1)

# p0 = np.sqrt(2*m*A*0.2)
# dp2 = p0 * p0 * 1. / 80
# d = hbar / np.sqrt(2 * dp2)


# Wave Function definitions
x0 = int(0.2 * x_length)
sig = 4
k_init = 5
psi_init = gauss_init(x, k_init, x0=x0, d=sig)

#
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

t = 0
dt = 0.01
step = 5
Ns = 600

print(dt * Ns * step)
sch = Schrodinger(x, psi_init, V_x, k, hbar=hbar, m=m, t=t)

################### Testing
# t_list = []
# for i in range(0, Ns):
#     t += dt * step
#     t_list.append(t)
# t = 0
#
# L_list = np.linspace(1, 6, 10)
# Tunnel_Val = []
#
# figure = plt.figure()
# ax = figure.add_subplot(1, 1, 1)
# ax.set_title("Tunneled Packet vs Time")
# ax.set_xlabel("Time")
# ax.set_ylabel("Probability of Particle Tunnelling")
# for a in L_list:
#     temp, t_list = run(a * dx, time=True,a=A)
#     Tunnel_Val.append(temp[-1])
#     ax.plot(t_list, temp, label='a={v}'.format(v=round(a, 3)))
#
# ax.legend()
# plt.savefig('Tunneling Stuff')
# plt.show()
#
# plt.plot(L_list, Tunnel_Val)
# plt.savefig('Tunneling Stuff 2')
# plt.show()

################# Initial print
# plt.plot(x, sch.mod_square_x(True))
# plt.plot(x, V_x)
# plt.show()

################ Animation
# a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(np.real(psi_init)))),
#             lim2=((min(ks), max(ks)), (0, 30)))
#
# a.make_fig()

################### Efficiency
# start = time.time()
# for i in range(Ns):
#     sch.evolve_t(step, dt)
# print(time.time()-start)

################### Multiple Amp
w_list = range(1, 20)
A_list = range(12, 20, 2)

# w_list = range(1, 15)
# A_list = [25]


mult = []
for a in A_list:
    save = [0 for k in range(len(w_list))]
    dat = run_w(w_list, a)
    plt.plot(w_list, dat, linestyle="", marker="o", label=str(a))
    # plt.loglog(w_list, dat, linestyle="", marker="o", label=str(a))

    for i in range(len(w_list)):
        save[i] = (w_list[i], dat[i])

    np.savetxt("Gauss_{a}.txt".format(a=a), save)

# np.savetxt("Log_test.txt", mult)

plt.title("Tunneling probability Vs Barrier Width")
plt.xlabel("Barrier width")
plt.ylabel("Tunneling probability")
plt.legend()
plt.savefig("Varying_Amplitude")

plt.show()
