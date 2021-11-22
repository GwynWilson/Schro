from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.integrate import simps


def hellerGaussian2D(x, y, vals, args):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


# matom = 1.44316072 * 10 ** -25
# Natom = 10 ** 6
# Natom = 1
# m = Natom * matom
# hbar = 1.0545718 * 10 ** -34
#
# N = 2 ** 8
# xlen = 0.1
# dx = xlen / N
#
# x = np.asarray([i * dx for i in range(N)]) - xlen / 2
# y = np.asarray([i * dx for i in range(N)]) - xlen / 2
# X, Y = np.meshgrid(x, y)
#
# k_lim = np.pi / dx
# k_arr = -k_lim + (2 * k_lim / N) * np.arange(N)
#
# xN = int(N / 2)
# yN = int(N / 2)
# x0 = x[xN]
# y0 = y[xN]
# vx0 = 0
# vy0 = 0.01
# print(m * vy0 / hbar)
# sigx = 10 ** -3
# sigy = 10 ** -3
# ax0 = 1j * hbar / (4 * sigx ** 2)
# ay0 = 1j * hbar / (4 * sigy ** 2)
# lam = 0
# g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# args = (m, hbar)
#
# t = 0
# tf = 5
# N_step = 100
# dt = tf / N_step
#
# psi_init = hellerGaussian2D(X, Y, init, args)
# psi_k = fftshift(fftn(psi_init))
# V = np.zeros((N, N))
#
# psi_kss = np.real(psi_k * np.conjugate(psi_k))
# fig, ax = plt.subplots()
# ax.imshow(psi_kss[:: -1])
# plt.show()
#
# sch = Schrodinger2D(x, y, psi_init, V, args)
# tl, xt, yt, xst, yst, = sch.expectationVals(N_step, dt)
#
# psi_final = sch.psi_x
#
# tool = PlotTools()
# tool.compareInit(psi_init, psi_final, xlen / 2, k_lim)

#################### Scaling
matom = 1.44316072 * 10 ** -25
Natom = 10 ** 6
Natom = 1
m = Natom * matom
hbar = 1.0545718 * 10 ** -34
t_tot = 100 * 10 ** (-6)
w = 1000
xlim = 4 * 10 ** (-5)
x0 = 0
y0 = 0
vx0 = 0
vy0 = 0.01

sig = np.sqrt(hbar / (2 * m * w))

ax0 = 1j * hbar / (4 * sig ** 2)
ay0 = 1j * hbar / (4 * sig ** 2)
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sig * sig)
init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar)

N = 2 ** 8
dx = xlim / N
x = np.asarray([i * dx for i in range(N)]) - xlim / 2
y = np.asarray([i * dx for i in range(N)]) - xlim / 2
X, Y = np.meshgrid(x, y)

k_lim = np.pi / dx
# print(k_lim > m * vy0 / hbar)

Nstep = 400
dt = t_tot / Nstep

psi_init = hellerGaussian2D(X, Y, init, args)

V = np.zeros((N, N))


########## Initial Plots
# psi_k = fftshift(fftn(psi_init))
# psi_s = np.real(psi_init * np.conjugate(psi_init))
# psi_kss = np.real(psi_k * np.conjugate(psi_k))
# fig, ax = plt.subplots(1, 2, figsize=(8, 5))
# ax[0].imshow(psi_s[:: -1], extent=[-xlim / 2, xlim / 2, -xlim / 2, xlim / 2])
# ax[1].imshow(psi_kss[::-1], extent=[-k_lim, k_lim, -k_lim, k_lim])
# plt.show()

############# Expectation Value
# sch = Schrodinger2D(x, y, psi_init, V, args)
# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep,dt)
#
# plt.plot(t_list,y_av)
# plt.show()
#
# tool = PlotTools()
# tool.compareInit(psi_init, sch.psi_x, xlim / 2, k_lim)


########## Realistic Ring Trap

def ring2D(x, y, args):
    m, hbar, w, r0 = args
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * m * w ** 2 * (r - r0) ** 2


r0 = 0.5 * 10 ** (-5)
x0 = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))
init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

args = (m, hbar, w, r0)
psi_init = hellerGaussian2D(X, Y, init, args)

V = ring2D(X, Y, args)
# fig, ax = plt.subplots()
# ax.imshow(V)
# plt.show()

sch = Schrodinger2D(x, y, psi_init, V, args)
tool = PlotTools()

# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt)
# tool.expectationValues(t_list,x_av,y_av,xs_av,ys_av)


tool.animation2D(sch, 1, 0.0001, args, xlim / 2, k_lim,save="Realistic_values")

############# Scaled Values
# m_s = m
# T_s = 2 * np.pi / w
# L_s = np.sqrt(hbar * T_s / m_s)
#
# m_p = m / m_s
# hbar_p = hbar * T_s / (m_s * L_s ** 2)
# t_tot_p = t_tot / T_s
# x_lim_p = xlim / L_s
# w_p = w * T_s
# x0_p = x0 / L_s
# y0_p = y0 / L_s
# vx0_p = vx0 * T_s / L_s
# vy0_p = vy0 * T_s / L_s
#
# sig_p = sig / L_s
#
# ax0_p = 1j * hbar_p / (4 * sig_p ** 2)
# ay0_p = 1j * hbar_p / (4 * sig_p ** 2)
# lam_p = 0
# g0_p = (1j * hbar_p / 2) * np.log(2 * np.pi * sig_p * sig_p)
# init_p = [x0_p, y0_p, vx0_p, vy0_p, ax0_p, ay0_p, lam_p, g0_p]
# args_p = (m_p, hbar_p)
#
# N = 2 ** 8
# dx_p = x_lim_p / N
# x_p = np.asarray([i * dx_p for i in range(N)]) - x_lim_p / 2
# y_p = np.asarray([i * dx_p for i in range(N)]) - x_lim_p / 2
# X_p, Y_p = np.meshgrid(x_p, y_p)
#
# k_lim_p = np.pi / dx_p
# print(k_lim_p > m_p * vy0_p / hbar_p)
#
# Nstep = 100
# dt_p = t_tot_p / Nstep
# psi_scaled = hellerGaussian2D(X_p, Y_p, init_p, args_p)

# Initial Plots
# psi_k = fftshift(fftn(psi_scaled))
# psi_s = np.real(psi_scaled * np.conjugate(psi_scaled))
# psi_kss = np.real(psi_k * np.conjugate(psi_k))
# fig, ax = plt.subplots(1, 2, figsize=(8, 5))
# ax[0].imshow(psi_s[:: -1], extent=[-x_lim_p / 2, x_lim_p / 2, -x_lim_p / 2, x_lim_p / 2])
# ax[1].imshow(psi_kss[::-1], extent=[-k_lim_p, k_lim_p, -k_lim_p, k_lim_p])
# plt.show()


# sch = Schrodinger2D(x_p, y_p, psi_scaled, V, args_p)
# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt_p)
#
# plt.plot(np.asarray(t_list)*T_s, np.asarray(y_av)*L_s)
# plt.show()

# tool = PlotTools()
# tool.compareInit(psi_scaled, sch.psi_x, x_lim_p / 2, k_lim_p)
# tool.animation2D(sch, 1, dt_p, args_p, x_lim_p, k_lim_p)
