import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from scipy.integrate import simps
import matplotlib.pyplot as plt

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * (x - x0))


def hellerPacket(sch, init):
    x0, p0, a0, g0, t0 = init
    xt = x0 + p0 * (sch.t - t0) / sch.m
    pt = p0
    at = a0 / ((2 / sch.m) * a0 * (sch.t - t0) + 1)
    gt = p0 ** 2 * (sch.t - t0) / (2 * sch.m) + 1j * sch.hbar / 2 * np.log(
        (2 / sch.m) * a0 * (sch.t - t0) + 1) + g0
    # gt = p0 ** 2 * (sch.t - t0) / (2 * sch.m) + g0
    psi_h = np.exp((1j / hbar) * at * (x - xt) ** 2 + (1j / hbar) * pt * (x - xt) + (1j / hbar) * gt)
    return xt, pt, at, gt, psi_h


def hellerInstance(x, xt, pt, at, gt):
    return np.exp((1j / hbar) * at * (x - xt) ** 2 + (1j / hbar) * pt * (x - xt) + (1j / hbar) * gt)


# Defining x axis
N = 2 ** 10
dx = 0.1
x_length = N * dx
x = np.asarray([i * dx for i in range(N)])
x0 = int(0.25 * x_length)

d = 1

# Defining Psi and V
k_initial = 1
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
Ns = 50
print("Final time", dt * step * Ns)

hbar = 1
m = 1

p0 = k_initial * hbar
a0 = 1j * hbar / (4 * d ** 2)
g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)
init = [x0, p0, a0, g0, t]

sch = Schrodinger(x, psi_x, V_x, k, hbar=hbar, m=m)

xt, pt, at, gt, psi_h = hellerPacket(sch, init)

# sch = Schrodinger(x, psi_h, V_x, k, hbar=hbar, m=m)

# fig, (ax1,ax2) = plt.subplots(1,2)
# ax1.plot(x,np.real(sch.psi_x))
# ax1.plot(x,np.real(psi_h),linestyle=":")
#
# ax2.plot(x,np.imag(sch.psi_x))
# ax2.plot(x,np.imag(psi_h),linestyle=":")
# plt.show()


############# Individual componenets
# t_list = []
# norm_x = []
# expec_x = []
# expec_xs = []
# expec_v = []
#
# norm_h = []
# hell_x = []
# hell_v = []
# hell_wid = []
#
# temp=[]
#
# for i in range(Ns):
#     if i != 0:
#         sch.evolve_t(step, dt)
#     t_list.append(sch.t)
#     norm_x.append(sch.norm_x() - 1)
#     expec_x.append(sch.expectation_x())
#     expec_xs.append(np.sqrt(sch.expectation_x_square() - expec_x[i] ** 2))
#     sch.normalise_k()
#     expec_v.append(sch.hbar * sch.expectation_k() / sch.m)
#
#     xt, pt, at, gt, psi_h = hellerPacket(sch, init)
#     hell_x.append(xt)
#     hell_v.append(pt / sch.m)
#
#     mod_psi_h = psi_h * np.conjugate(psi_h)
#     norm_h.append(np.real(simps(mod_psi_h, sch.x)) - 1)
#
#     xs_h = simps(mod_psi_h * sch.x ** 2, sch.x)
#     hell_wid.append(np.sqrt(xs_h - xt ** 2))
#
#     temp.append(1j/hbar *(at-np.conjugate(at)))
#
# plt.plot(t_list, norm_x, linestyle='none', marker='x', label="Fourier Method")
# plt.plot(t_list, norm_h, linestyle='none', marker='x',label="Heller")
# plt.title('Normalistaion of wavefunction over time')
# plt.xlabel('Time')
# plt.ylabel('Normalisation-1')
# plt.legend(loc='best', fancybox=True)
# plt.show()
#
# plt.plot(t_list, expec_x, label="Fourier Method")
# plt.plot(t_list, hell_x, linestyle='--', label='Heller')
# plt.title('Expectation value of x over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<x>$')
# plt.legend(loc='best', fancybox=True)
# plt.show()
#
# plt.plot(t_list, expec_v, label="Fourier Method")
# plt.plot(t_list, hell_v, linestyle='--', label='Heller')
# plt.title('Expectation value of v over time')
# plt.xlabel('Time')
# plt.ylabel(r'$<v>$')
# plt.ylim((0,10))
# plt.legend(loc='best', fancybox=True)
# plt.show()
#
# plt.plot(t_list, np.asarray(expec_xs), label="Fourier Method")
# plt.plot(t_list, hell_wid, linestyle='--', label='Heller')
# plt.plot(t_list,1/-np.asarray(temp))
# plt.title(r'Expectation value of $\Delta x$ over time')
# plt.xlabel('Time')
# plt.ylabel(r'$\Delta x$')
# plt.legend(loc='best', fancybox=True)
# plt.show()

#############Animation
# sch2 = Schrodinger(x, psi_h, V_x, k, hbar=hbar, m=m)
#
# a = Animate(sch, V_x, step, dt, lim1=((0, 0.75*x_length), (0, 0.4)),
#             lim2=((ks[0], ks[N - 1]), (0, 0.5)))
# a.make_fig(func=hellerPacket,args=init,label="Heller")


########### Psi_h comp
t_list = []
squared_list = []
for i in range(Ns):
    if i != 0:
        sch.evolve_t(step, dt)
    t_list.append(sch.t)
    psi_sol = sch.psi_x
    xt, pt, at, gt, psi_h = hellerPacket(sch, init)
    squared = np.conjugate(psi_sol)*psi_h
    squared_list.append(simps(squared,sch.x))

plt.plot(t_list,np.asarray(squared_list)-1)
plt.title("Overlap of Heller and Split step method")
plt.xlabel("Time (t)")
plt.ylabel(r"$\langle \psi_s (x) \vert \psi_h (x) \rangle$ -1")
plt.savefig("Heller Overlap")
plt.show()
