import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D


def gauss1D(x, x0, sig):
    return 1 / (2 * np.pi * sig ** 2) ** (1 / 4) * np.exp(-(x - x0) ** 2 / (4 * sig ** 2))


def hellerGaussian2D(x, y, vals, args):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def initialGaussian(x, y, x0, y0, sigx, sigy):
    return 1 / np.sqrt(2 * np.pi * sigx * sigy) * np.exp(-(x - x0) ** 2 / (4 * sigx ** 2)) * np.exp(
        -(y - y0) ** 2 / (4 * sigy ** 2))


def ftInitialGaussian(kx, ky, x0, y0, sigx, sigy):
    # return np.sqrt(8 * np.pi * sigx * sigy) * np.exp(-sigx ** 2 * kx ** 2 - 1j * x0 * kx) * np.exp(
    #     -sigy ** 2 * ky ** 2 - 1j * y0 * ky)
    return 1 / (np.sqrt(4 * np.pi ** 2)) * np.sqrt(8 * np.pi * sigx * sigy) * np.exp(
        -sigx ** 2 * kx ** 2 - 1j * x0 * kx) * np.exp(
        -sigy ** 2 * ky ** 2 - 1j * y0 * ky)


def slope(x, y, args):
    m, hbar, al, be = args
    return m * al * x + m * be * y


def derivsDtNone(t, current, args, eta, dt):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = current
    xn = vxt * dt
    yn = vyt * dt
    vxn = 0
    vyn = 0
    axn = (-2 * axt ** 2 / m - 0.5 * lamt ** 2 / m) * dt
    ayn = (-2 * ayt ** 2 / m - 0.5 * lamt ** 2 / m) * dt
    lamn = (-2 * (axt + ayt) * lamt / m) * dt
    gn = (1j * hbar * axt / m + 1j * hbar * ayt / m + 0.5 * m * vxt ** 2 + 0.5 * m * vyt ** 2) * dt
    return xn, yn, vxn, vyn, axn, ayn, lamn, gn


def derivsDtSlope(t, current, args, eta, dt):
    m, hbar, al, be = args
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = current
    xn = vxt * dt
    yn = vyt * dt
    vxn = -al * dt
    vyn = -be * dt
    axn = (-2 * axt ** 2 / m - 0.5 * lamt ** 2 / m) * dt
    ayn = (-2 * ayt ** 2 / m - 0.5 * lamt ** 2 / m) * dt
    lamn = (-2 * (axt + ayt) * lamt / m) * dt
    gn = (1j * hbar * axt / m + 1j * hbar * ayt / m + 0.5 * m * vxt ** 2 + 0.5 * m * vyt ** 2) * dt
    return xn, yn, vxn, vyn, axn, ayn, lamn, gn


def hellerValues(N_step, dt, func):
    hel = Heller2D(N_step, dt, init, func)
    tl, hel_arr = hel.rk4(args)
    t_list = [i * dt for i in range(N_step)]
    xh_av = np.zeros(N_step)
    yh_av = np.zeros(N_step)
    xhs_av = np.zeros(N_step)
    yhs_av = np.zeros(N_step)
    for n, v in enumerate(tl):
        psi_h = hellerGaussian2D(X, Y, hel_arr[:, n], args)
        psi_hs = np.real(psi_h * np.conjugate(psi_h))
        xh_av[n] = simps(simps(X * psi_hs, x), y)
        yh_av[n] = simps(simps(Y * psi_hs, y), x)
        xhs_av[n] = simps(simps(X ** 2 * psi_hs, x), y)
        yhs_av[n] = simps(simps(Y ** 2 * psi_hs, y), x)

    return t_list, xh_av, yh_av, xhs_av, yhs_av


def plotComparisson(psi1, psi2, xlen, klen):
    psi1k = fftshift(fftn(psi1))
    Z1 = np.real(psi1 * np.conjugate(psi1))
    Z1k = np.real(psi1k * np.conjugate(psi1k))

    psi2k = fftshift(fftn(psi2))
    Z2 = np.real(psi2 * np.conjugate(psi2))
    Z2k = np.real(psi2k * np.conjugate(psi2k))

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    data = ax[0].imshow(Z1, extent=[-xlen, xlen, -xlen, xlen])
    ax[1].imshow(Z2, extent=[-xlen, xlen, -xlen, xlen])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    data = ax[0].imshow(Z1k, extent=[-klen, klen, -klen, klen])
    ax[1].imshow(Z2k, extent=[-klen, klen, -klen, klen])
    plt.tight_layout()
    plt.show()


N = 500
len = 200
dx = len / N

x = np.asarray([i * dx for i in range(N)]) - len / 2
y = np.asarray([i * dx for i in range(N)]) - len / 2
# y = y[::-1]  # Reversing array for easier computations

hbar = 1
m = 1

args = (m, hbar)

x0 = 0
y0 = 0

vx0 = 0
vy0 = 0

sigx = 8
sigy = 8

ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)

lam = 0

g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

X, Y = np.meshgrid(x, y)
psi = hellerGaussian2D(X, Y, init, args)

Z = np.real(psi * np.conjugate(psi))

##### Plot of psi^2
# psi_init = initialGaussian(X, Y, x0, y0, sigx, sigy)
# Z_inital = psi_init * np.conjugate(psi_init)
#
# fig, ax = plt.subplots()
# fig.suptitle("Comparision of initial Gaussian")
# data = ax.imshow(Z_inital - Z, extent=[-len / 2, len / 2, -len / 2, len / 2])
# cb = fig.colorbar(data)
# plt.show()
#
# fig, ax = plt.subplots()
# data = ax.imshow(Z, extent=[-len / 2, len / 2, -len / 2, len / 2])
# cb = fig.colorbar(data)
# plt.show()


#### Normalisation
# norm = simps(simps(Z,x),y)


### FFT
# psik = fftn(psi)
# psik = fftshift(psik)
#
# k_lim = np.pi / dx
# k_arr = -k_lim + (2 * k_lim / N) * np.arange(N)
#
# bac = ifftn(psik)
#
# Z = np.real(psik * np.conjugate(psik))
# norm = simps(simps(Z,k_arr),k_arr)
# Z=Z/norm
#
# fig, ax = plt.subplots()
# data = ax.imshow(Z, extent=[-k_lim, k_lim, -k_lim, k_lim])  # The ::-1 is to flip the axes
# cb = fig.colorbar(data)
# plt.show()

# KX, KY = np.meshgrid(k_arr, k_arr)
#
# psik_init = ftInitialGaussian(KX, KY, x0, y0, sigx, sigy)
# Z_init = np.real(psik_init * np.conjugate(psik_init))
#
# fig, ax = plt.subplots()
# data = ax.imshow(Z_init[::-1], extent=[-k_lim, k_lim, -k_lim, k_lim])  # The ::-1 is to flip the axes
# cb = fig.colorbar(data)
# plt.show()


####### Sim Test k Flat pot
# t = 0
# ttot = 50
# N_step = 100
# dt = ttot / N_step
#
# V = np.zeros((N, N))  # Flat potential
#
# sch = Schrodinger2D(x, y, psi, V, args, t=t)
# sch.evolvet(N_step, dt)
#
# psif = sch.psi_x
# Zf = np.real(psif * np.conjugate(psif))
#
# fig, ax = plt.subplots()
# data = ax.imshow(Zf, extent=[-len / 2, len / 2, -len / 2, len / 2])
# cb = fig.colorbar(data)
# plt.show()


####### Sim harmonic
# x0 = 20
# w = 0.01
#
# T = 2 * np.pi * w
# args = (m, hbar, w)
# print(T)
#
# V = slopex(X, Y, args)
#
# # Potential plot
# fig, ax = plt.subplots()
# data = ax.imshow(V, extent=[-len / 2, len / 2, -len / 2, len / 2])
# cb = fig.colorbar(data)
# plt.show()
#
# t = 0
# nstep = 100
# ttot = 100
# dt = ttot / nstep
#
# x0 = 25
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# psi = hellerGaussian2D(X, Y, init, args)
#
# k_lim = np.pi / dx
#
# sch = Schrodinger2D(x, y, psi, V, args)
# sch.evolvet(nstep, dt)
# psif = sch.psi_x
#
# plotComparisson(psi, psif, len / 2, k_lim)

######## Expectation values
# x0 = x[int(3 * N / 4)]
# Nx = int(3 * N / 4)
# t = 0
# N_step = 100
# Tf = 100
# w = 0.01
# dt = Tf / N_step
# args = (m, hbar, w)
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# psi = hellerGaussian2D(X, Y, init, args)
#
# V = slopex(x, y, args)
# V = np.zeros((N,N))
# sch = Schrodinger2D(x, y, psi, V, args)
#
# Z = np.real(psi * np.conjugate(psi))
#
# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(N_step, dt)
#
# x_traj = -0.25 * w * np.asarray(t_list) ** 2 + x0
# x_traj = np.zeros(N_step) +x0
# y_traj = np.zeros(N_step)
# widthx = sigx * np.sqrt(1 + np.asarray(t_list) ** 2 * (hbar / (2 * m * sigx ** 2)))
# widthy = sigy * np.sqrt(1 + np.asarray(t_list) ** 2 * (hbar / (2 * m * sigy ** 2)))
#
# tool = PlotTools()
# # tool.expectationValues(t_list, x_av, y_av, xs_av, ys_av)
# tool.expectationValuesComparrison(t_list, x_av, y_av, xs_av, ys_av, x_traj, y_traj, widthx, widthy)


############ Heller Comparrison
# t = 0
# ttot = 100
# N_step = 100
# dt = ttot / N_step
#
# V = np.zeros((N, N))  # Flat potential
# psi_i = psi
#
# sch = Schrodinger2D(x, y, psi, V, args, t=t)
# # sch.evolvet(N_step, dt)
# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(N_step, dt)
#
# psif = sch.psi_x
# Zf = np.real(psif * np.conjugate(psif))
# k_lim = np.pi / dx
#
# tool = PlotTools()
# tool.compareInit(psi_i, psif, len / 2, k_lim)
#
# t_list, xh_av, yh_av, xhs_av, yhs_av = hellerValues(N_step, dt, derivsDtNone)
#
# x_traj = xh_av
# y_traj = yh_av
# widthx = np.sqrt(xhs_av - xh_av ** 2)
# widthy = np.sqrt(yhs_av - yh_av ** 2)
#
# tool.expectationValuesComparrison(t_list, x_av, y_av, xs_av, ys_av, x_traj, y_traj, widthx, widthy)

######### Comparisson slope
# x0 = x[int(3 * N / 4)]
# y0 = y[int(3 * N / 4)]
# Nx = int(3 * N / 4)
# t = 0
# N_step = 500
# Tf = 100
# al = 0.01
# be = 0.01
# dt = Tf / N_step
# args = (m, hbar, al, be)
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# psi = hellerGaussian2D(X, Y, init, args)
# psi_i = psi
#
# V = slope(X, Y, args)
# fig,ax=plt.subplots()
# dat = ax.imshow(V[::-1],extent=[-len/2,len/2,-len/2,len/2])
# cb = fig.colorbar(dat)
# fig.suptitle("Potential")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.savefig("Slope_Potential")
# plt.show()
#
# sch = Schrodinger2D(x, y, psi, V, args, t=t)
# # sch.evolvet(N_step, dt)
# t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(N_step, dt)
#
# psif = sch.psi_x
# Zf = np.real(psif * np.conjugate(psif))
# k_lim = np.pi / dx
#
# tool = PlotTools()
# tool.compareInit(psi_i, psif, len / 2, k_lim,save="Slope")
#
# t_list, xh_av, yh_av, xhs_av, yhs_av = hellerValues(N_step, dt, derivsDtSlope)
#
# x_traj = xh_av
# y_traj = yh_av
# widthx = np.sqrt(xhs_av - xh_av ** 2)
# widthy = np.sqrt(yhs_av - yh_av ** 2)
#
# fig, ax = tool.expectationValuesComparrison(t_list, x_av, y_av, xs_av, ys_av, x_traj, y_traj, widthx, widthy,
#                                             show=False)
# fig.suptitle("Schrodinger - Heller Comparison")
# ax[0, 0].lines[0].set_label("Split-Step")
# ax[0, 0].lines[1].set_label("Heller")
# ax[0, 0].legend()
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("Slope_Comparison")
# plt.show()
