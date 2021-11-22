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
    tool.animateComparrison(sch, Nstep, tl, hel_arr, args, xlen / 2, k_lim, full=False,save="Test")


N = 400
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
sigx = 8
sigy = 8
ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)
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
T_tot = 4 * np.pi / w
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
r0 = 50
w=0.02

sig = np.sqrt(hbar / (2 * m * w))
print(sig)

x0 = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))
print(x0,xlen)
ax0 = 1j * hbar / (4 * sig ** 2)
ay0 = 1j * hbar / (4 * sig ** 2)
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sig * sig)
init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

psi = hellerGaussian2D(X,Y,init,args)

Nstep = 400
dt = np.pi / 2
runComp(psi, V)
# runSim(psi,V)
