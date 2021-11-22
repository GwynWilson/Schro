from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.integrate import simps


def harmonic2D(x, y, args):
    m, hbar, w1, w2 = args
    return 0.5 * m * w1 ** 2 * x ** 2 + 0.5 * m * w2 ** 2 * y ** 2


def hellerGaussian2D(x, y, vals, args):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def derivsdt(t, current, args, eta, dt):
    m, hbar, w1, w2 = args
    x = current[2] * dt
    y = current[3] * dt
    vx = -m * w1 ** 2 * current[0] * dt
    vy = -m * w2 ** 2 * current[1] * dt
    ax = (-(2 / m) * current[4] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w1 ** 2) * dt
    ay = (-(2 / m) * current[5] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w2 ** 2) * dt
    lam = (-2 * ((current[4] / m) + (current[5] / m)) * current[6]) * dt
    gam = (1j * hbar * current[4] / m + 1j * hbar * current[5] / m + 0.5 * m * current[2] ** 2 + 0.5 * m * current[
        3] ** 2 - 0.5 * m * w1 ** 2 * current[0] ** 2 - 0.5 * m * w2 ** 2 * current[1] ** 2) * dt
    return x, y, vx, vy, ax, ay, lam, gam


def hellerValues(N_step, dt, func, args):
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


def plotPotential(X, Y, xlen, ylen, args):
    fig, ax = plt.subplots()
    dat = ax.imshow(harmonic2D(X, Y, args)[::-1], extent=[-xlen, xlen, -ylen, ylen])
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


def runSim(psi, V):
    sch = Schrodinger2D(x, y, psi, V, args)
    # sch.evolvet(Nstep, dt)
    t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt)
    psif = sch.psi_x

    t_l, xh_av, yh_av, xhs_av, yhs_av = hellerValues(Nstep, dt, derivsdt, args)

    tool = PlotTools()
    # tool.compareInit(psi, psif, xlen, k_lim)
    # tool.expectationValues(t_list, x_av, y_av, xs_av, ys_av)
    # tool.expectationValues(t_list, xh_av, yh_av, xhs_av, yhs_av)

    xh_wid = np.sqrt(xhs_av - xh_av ** 2)
    yh_wid = np.sqrt(yhs_av - yh_av ** 2)
    fig, ax = tool.expectationValuesComparrison(t_list, x_av, y_av, xs_av, ys_av, xh_av, yh_av, xh_wid, yh_wid,
                                                show=False)
    fig.suptitle(f"$Schrodinger - Heller Comparison,\omega_x={w1} ,\omega_y={w2}$")
    ax[0, 0].lines[0].set_label("Split-Step")
    ax[0, 0].lines[1].set_label("Heller")
    ax[0, 0].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig("HO")
    plt.show()


N = 200
xlen = 200
dx = xlen / N

x = np.asarray([i * dx for i in range(N)]) - xlen / 2
y = np.asarray([i * dx for i in range(N)]) - xlen / 2
X, Y = np.meshgrid(x, y)

k_lim = np.pi / dx
k_arr = -k_lim + (2 * k_lim / N) * np.arange(N)

m = 1
hbar = 1

w1 = 0.02  # set to stay within k range
w2 = 0.02

xN = int(3 * N / 4)
yN = int(5 * N / 8)
x0 = x[xN]
y0 = y[xN]
vx0 = 0
vy0 = 0
sigx = 8
sigy = 8
ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar, w1, w2)

t = 0
T_tot = 2 * np.pi / w1
Nstep = 500
dt = T_tot / Nstep

# plotPotential(X,Y,xlen,xlen,args) #Potential plot
# plotInitial(X, Y, init, args, xlen, xlen) #Initial Plot x space


######## Running Simulation
# psi = hellerGaussian2D(X, Y, init, args)
#
# V = harmonic2D(X, Y, args)
# runSim(psi, V)

###### Circular Orbits
# y0 = 0
# vy0 = 1
# x0 = vy0 / w1
#
# t = 0
# T_tot = 2 * np.pi / w1
# Nstep = 500
# dt = T_tot / Nstep
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# psi = hellerGaussian2D(X, Y, init, args)
#
# V = harmonic2D(X, Y, args)
# runSim(psi, V)


##### Wobbling

y0 = 0
x0 = 0
ax0 = 1j * m*w1/2
ay0 = 1j * m*w1/2


t = 0
T_tot = 2 * np.pi / w1
Nstep = 500
dt = T_tot / Nstep


init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
psi = hellerGaussian2D(X, Y, init, args)

V = harmonic2D(X, Y, args)
runSim(psi, V)