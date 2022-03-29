from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D, PlotTools2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.integrate import simps


def ring2D(x, y, args):
    m, hbar, w, r0 = args
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * m * w ** 2 * (r - r0) ** 2


def hellerGaussian2D(x, y, vals, args):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def plotPotential(X, Y, xlen, ylen, args):
    fig, ax = plt.subplots()
    dat = ax.imshow(ring2D(X, Y, args)[::-1], extent=[-xlen, xlen, -ylen, ylen])
    fig.colorbar(dat)
    plt.show()

    psi = hellerGaussian2D(X, Y, init, args)
    Z = np.real(psi * np.conjugate(psi))

    print(simps(simps(Z, x), y))

    fig, ax = plt.subplots()
    dat = ax.imshow(Z[::-1], extent=[-xlen, xlen, -ylen, ylen])
    fig.colorbar(dat)
    plt.show()


def plotInitial(X, Y, init, args, xlen, ylen):
    psi = hellerGaussian2D(X, Y, init, args)
    Z = np.real(psi * np.conjugate(psi))

    print(simps(simps(Z, x), y))

    fig, ax = plt.subplots()
    dat = ax.imshow(Z[::-1], extent=[-xlen, xlen, -ylen, ylen])
    fig.colorbar(dat)
    plt.show()


def hellerValues(N_step, dt, func, args, arr=False):
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

    if arr:
        return t_list, xh_av, yh_av, xhs_av, yhs_av, hel_arr
    else:
        return t_list, xh_av, yh_av, xhs_av, yhs_av


def derivsDt(t, current, args, eta, dt):
    m, hbar, w, r0 = args
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = current
    xn = vxt * dt
    yn = vyt * dt
    vxn = (-w ** 2 * xt * (1 - r0 / (xt ** 2 + yt ** 2) ** (1 / 2))) * dt
    vyn = (-w ** 2 * yt * (1 - r0 / (xt ** 2 + yt ** 2) ** (1 / 2))) * dt
    axn = (-2 * axt ** 2 / m - 0.5 * lamt ** 2 / m - 0.5 * m * w ** 2 * (
            1 - r0 * yt ** 2 / (xt ** 2 + yt ** 2) ** (3 / 2))) * dt
    ayn = (-2 * ayt ** 2 / m - 0.5 * lamt ** 2 / m - 0.5 * m * w ** 2 * (
            1 - r0 * xt ** 2 / (xt ** 2 + yt ** 2) ** (3 / 2))) * dt
    lamn = (-2 * (axt + ayt) * lamt / m - m * w ** 2 * r0 * xt * yt / (xt ** 2 + yt ** 2) ** (3 / 2)) * dt
    gn = (1j * hbar * axt / m + 1j * hbar * ayt / m + 0.5 * m * vxt ** 2 + 0.5 * m * vyt ** 2 - 0.5 * m * w ** 2 * (
            (xt ** 2 + yt ** 2) ** (1 / 2) - r0) ** 2) * dt
    return xn, yn, vxn, vyn, axn, ayn, lamn, gn


def runSim(psi, V):
    sch = Schrodinger2D(x, y, psi, V, args)
    # sch.evolvet(Nstep, dt)
    t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt)
    psif = sch.psi_x

    t_l, xh_av, yh_av, xhs_av, yhs_av = hellerValues(Nstep, dt, derivsDt, args)

    tool = PlotTools()
    # tool.compareInit(psi, psif, xlen, k_lim)
    # tool.expectationValues(t_list, x_av, y_av, xs_av, ys_av)
    # tool.expectationValues(t_list, xh_av, yh_av, xhs_av, yhs_av)

    xh_wid = np.sqrt(xhs_av - xh_av ** 2)
    yh_wid = np.sqrt(yhs_av - yh_av ** 2)
    fig, ax = tool.expectationValuesComparrison(t_list, x_av, y_av, xs_av, ys_av, xh_av, yh_av, xh_wid, yh_wid,
                                                show=False)
    fig.suptitle(f"$Schrodinger - Heller Comparison$")
    ax[0, 0].lines[0].set_label("Split-Step")
    ax[0, 0].lines[1].set_label("Heller")
    ax[0, 0].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Ring Comparrison")
    plt.show()


def runSch(psi):
    sch = Schrodinger2D(x, y, psi, V, args)
    # sch.evolvet(Nstep, dt)
    t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt)
    psif = sch.psi_x

    tool = PlotTools()
    tool.compareInit(psi, psif, xlen / 2, k_lim)
    fig, ax = tool.expectationValues(t_list, x_av, y_av, xs_av, ys_av, show=False)

    fig.suptitle(f"Schrodinger")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig("HO")
    plt.show()


def runHel():
    hel = Heller2D(Nstep, dt, init, derivsDt)
    tl, hel_arr = hel.rk4(args)
    # tool = PlotTools2D(tl, hel_arr, xlen / 2, xlen / 2, N)
    # tool.animation2D(args)
    tool = PlotTools()
    tool.hellerAnimation(Nstep, X, Y, tl, hel_arr, args, xlen / 2)


def runComp(psi, V):
    sch = Schrodinger2D(x, y, psi, V, args)
    hel = Heller2D(Nstep, dt, init, derivsDt)
    tl, hel_arr = hel.rk4(args)
    tool = PlotTools()

    tool.animateComparrison(sch, Nstep, tl, hel_arr, args, xlen / 2, k_lim, full=False,rp=x0,save="Test")
    tool.animateComparrison(sch, Nstep, tl, hel_arr, args, xlen / 2, k_lim, full=False, rp=x0)

def runCompFinal(psi, V):
    # sch = Schrodinger2D(x, y, psi, V, args)
    # sch.evolvet(Nstep,dt)
    #
    # hel = Heller2D(Nstep, dt, init, derivsDt)
    # tl, hel_arr = hel.rk4(args)
    # psi_sch = sch.psi_x
    # # print(hel_arr[-1])
    # psi_hel = hellerGaussian2D(X,Y,hel_arr[:,-1],args)
    #
    #
    # # Z1 = np.real(sch.psi_x*np.conjugate(sch.psi_x))
    # # Z2 = np.real(psi_hel*np.conjugate(psi_hel))
    # save = np.savez("Comparrison", sch=sch.psi_x,hel=psi_hel)

    loaded = np.load("Comparrison.npz")
    psi_sch = loaded["sch"]
    psi_hel = loaded["hel"]

    Z1 = np.real(psi_sch*np.conjugate(psi_sch))
    Z2 = np.real(psi_hel*np.conjugate(psi_hel))

    x_expect = simps(simps(X * Z1, x), y)
    y_expect = simps(simps(Y * Z1, y), x)
    #
    # KX, KY = np.meshgrid(k_arr, k_arr)
    # psik_sch = fftshift(fftn(psi_sch))
    # psik_sch *= np.exp(1j * x_expect * KX)
    # psik_sch *= np.exp(1j * y_expect * KY)
    # psi_sch = ifftn(fftshift(psik_sch))
    #
    # psik_hel = fftshift(fftn(psi_hel))
    # psik_hel *= np.exp(1j * x_expect * KX)
    # psik_hel *= np.exp(1j * y_expect * KY)
    # psi_hel = ifftn(fftshift(psik_hel))
    #
    # Z1 = np.real(psi_sch*np.conjugate(psi_sch))
    # Z2 = np.real(psi_hel*np.conjugate(psi_hel))


    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    theta = np.linspace(-np.pi, np.pi, N)
    thetax = x0 * np.cos(theta)
    thetay = x0 * np.sin(theta)
    thetadup = (thetax, thetay)

    plotlim = 8 * sigx
    ax[0].imshow(Z1[::-1], extent=[-xlen/2, xlen/2, -xlen/2, xlen/2])
    ax[1].imshow(Z2[::-1], extent=[-xlen/2, xlen/2, -xlen/2, xlen/2])

    # ax[0].plot(thetax - x_expect, thetay - y_expect, linestyle="--", color="w")
    # ax[1].plot(thetax - x_expect, thetay - y_expect, linestyle="--", color="w")
    ax[0].plot(thetax, thetay, linestyle="--", color="w")
    ax[1].plot(thetax, thetay, linestyle="--", color="w")
    ax[0].set_title("Split-Step")
    ax[1].set_title("Heller")
    ax[0].set_xlim(-plotlim+x_expect, plotlim+x_expect)
    ax[0].set_ylim(-plotlim+y_expect, plotlim+y_expect)
    ax[1].set_xlim(-plotlim+x_expect, plotlim+x_expect)
    ax[1].set_ylim(-plotlim+y_expect, plotlim+y_expect)
    plt.savefig("Sch_Hel_Comp")
    plt.show()

N = 1000
xlen = 250
dx = xlen / N

x = np.asarray([i * dx for i in range(N)]) - xlen / 2
y = np.asarray([i * dx for i in range(N)]) - xlen / 2
X, Y = np.meshgrid(x, y)

k_lim = np.pi / dx
k_arr = -k_lim + (2 * k_lim / N) * np.arange(N)

m = 1
hbar = 1

xN = int(3 * N / 4)
w = 0.02  # set to stay within k range
r0 = x[xN]

x0 = 0
y0 = 0
vx0 = 0
vy0 = 0
# sigx = 8
# sigy = 8
# ax0 = 1j * hbar / (4 * sigx ** 2)
# ay0 = 1j * hbar / (4 * sigy ** 2)
ax0 = 1j * m * w / 2
ay0 = 1j * m * w / 2
sigx = np.sqrt(hbar / (2 * m * w))
sigy = np.sqrt(hbar / (2 * m * w))
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar, w, r0)

##### Circular orbit off
# x0 = r0
# vy0 = x0 / w
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
#
# t = 0
# T_tot = 2 * np.pi / w
# Nstep = 100
# dt = T_tot / Nstep
#
# Nstep = 1
# dt = 3
#
# V = ring2D(X, Y, args)
# psi = hellerGaussian2D(X, Y, init, args)
#
# # plotPotential(X, Y, xlen / 2, xlen / 2, args)
# # plotInitial(X, Y, init, args, xlen / 2, xlen / 2)
# # runSch(psi)
#
# sch = Schrodinger2D(x, y, psi, V, args)
#
# tool = PlotTools()
# tool.animation2D(sch, Nstep, dt, args, xlen / 2, k_lim, save="Test")


##### Circular orbit exact
x0 = r0
vy0 = 1

x0 = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))

# x0 = 70

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

t = 0
T_tot = 8 * np.pi / w
# T_tot = 358
Nstep = 400
dt = T_tot / Nstep

# Nstep = 1
# dt = 3

V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)

# plotPotential(X, Y, xlen / 2, xlen / 2, args)
# plotInitial(X, Y, init, args, xlen / 2, xlen / 2)
# runSch(psi)
# runSim(psi,V)
# runHel()


########### 2D Animation
# sch = Schrodinger2D(x, y, psi, V, args)
# tool = PlotTools()
# tool.animation2D(sch, Nstep, dt, args, xlen / 2, k_lim,save="Test")


########### Sch Hel Comparrison
x0 = r0
vy0 = 1

x0 = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))

# print(x0,xlen)
ax0 = 1j * m * w / 2
ay0 = 1j * m * w / 2
sigx = np.sqrt(hbar / (2 * m * w))
sigy = np.sqrt(hbar / (2 * m * w))
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)
init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

psi = hellerGaussian2D(X, Y, init, args)

Nstep = 800
dt = np.pi / 4
# runComp(psi, V)
# runSim(psi,V)
runCompFinal(psi,V)
