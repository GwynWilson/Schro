from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D, PlotTools2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fftpack import fftn, ifftn, fftshift


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


def freeAnimate(Nstep, dt):
    sch = Schrodinger2D(x, y, psi, V, args)
    tool = PlotTools()
    tool.animation2D(sch, Nstep, dt, args, xlen / 2, k_lim)


def expectation(sch, Nstep, dt):
    t_list, x_av, y_av, xs_av, ys_av = sch.expectationVals(Nstep, dt)
    tool = PlotTools()
    tool.expectationValues(t_list, x_av, y_av, xs_av, ys_av)


def comparrison(sch, Nstep, dt):
    t_l, x_s, y_s, xs_s, ys_s = sch.expectationVals(Nstep, dt)

    t_l, xh, yh, xhs, yhs = hellerValues(Nstep, dt, derivsDt, args)
    widx = np.sqrt(xhs - xh ** 2)
    widy = np.sqrt(yhs - yh ** 2)

    tool = PlotTools()
    tool.expectationValuesComparrison(t_l, x_s, y_s, xs_s, ys_s, xh, yh, widx, widy)


def getRing(theta, psis, A, xt, yt, shift=False):
    xring = A * np.cos(theta)
    # print(xring,X.shape)
    yring = A * np.sin(theta)

    psi_list = []
    thet = []
    for xpar, ypar in zip(xring, yring):
        indx = np.abs(x - xpar).argmin()
        # indy = np.abs(y[::-1] - ypar).argmin()  # List is reversed since meshgrid is reversed
        indy = np.abs(y - ypar).argmin()
        # print(xpar,x[indx],"\t\t",ypar,y[::-1][indy])
        # print(X[indy,indx],xpar,"\t\t",Y[indy,indx],ypar)
        psi_list.append(psis[indy, indx])

        # thet.append(np.arctan2(ypar,xpar))
        # # thet.append(np.arctan(ypar/xpar))

    # if shift:
    #     angle = np.arctan2(yt, xt)
    #     if abs(angle) > np.pi/2:
    #         print("Here")
    #         plt.plot(theta,psi_list)
    #         fftshift(psi_list)
    #         plt.plot(fftshift(theta),psi_list)
    #         plt.show()

    return psi_list


N = 800
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

### Circular Orbit
x0 = r0
vy0 = 1
rp = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))
x0 = rp
y0 = 0

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar, w, r0)

t = 0
T_tot = 4 * np.pi / w
Nstep = 500
dt = T_tot / Nstep

V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)

psis = np.real(psi * np.conjugate(psi))
# fig, ax = plt.subplots()
# ax.imshow(psis[::-1])
# plt.show()

sch = Schrodinger2D(x, y, psi, V, args)
# freeAnimate(1,3)
# expectation(sch, Nstep, dt)
# comparrison(sch, Nstep, dt)

# print(np.arctan(x0/y0)/np.pi)

############# Theta Testing
# theta = np.linspace(-np.pi, np.pi, N)
# # theta = fftshift(theta)
# # theta = np.linspace(0, 2 * np.pi, N)
# psi_ring = getRing(theta, psis, rp, x0, y0)
#
# angle1 = np.arctan2(y0, x0)
# theta1 = theta - angle1
# plt.plot(theta, psi_ring * theta1)
# plt.show()
# theta_ex = simps(psi_ring * theta1, theta1)
# theta_s_ex = simps(psi_ring * theta1 ** 2, theta1)
# print("Expected Val", theta_ex, "\t\t Expected Width", np.sqrt(theta_s_ex - theta_ex ** 2))
#
# init2 = [x0, -y0, vx0, vy0, ax0, ay0, lam, g0]
# psi = hellerGaussian2D(X, Y, init2, args)
# psis = np.real(psi * np.conjugate(psi))
#
# psi_ring2 = getRing(theta, psis, rp, x0, -y0)
# angle2 = np.arctan2(-y0, x0)
# theta2 = theta - angle2
# # theta2 = fftshift(theta)
# theta_ex = simps(psi_ring2 * theta2, theta2)
# theta_s_ex = simps(psi_ring2 * theta2 ** 2, theta2)
# print("Expected Val 2", theta_ex, "\t\t Expected Width 2", np.sqrt(theta_s_ex - theta_ex ** 2))


############## Schdo
# tl, theta_av, theta_wid = sch.ringExpectation(Nstep, dt, rp)
# plt.plot(tl,theta_av)
# plt.show()
#
# plt.plot(tl,theta_wid)
# plt.show()


############# Animate
# Nstep=1
# tool = PlotTools()
# tool.animateRing(sch, Nstep, dt, args, xlen / 2, k_lim, rp,save="Test")


############ Animate correct width
# ax0 = 1j * m*w/2
# ay0 = 1j * m*w/2
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# V = ring2D(X, Y, args)
# psi = hellerGaussian2D(X, Y, init, args)
# sch = Schrodinger2D(x, y, psi, V, args)
#
# Nstep=1
# tool = PlotTools()
# tool.animateRing(sch, Nstep, dt, args, xlen / 2, k_lim, rp,save="Test")


########### Animate Off Centre
x0 = x0-7
ax0 = 1j * m*w/2
ay0 = 1j * m*w/2

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)
sch = Schrodinger2D(x, y, psi, V, args)

Nstep=1
tool = PlotTools()
# tool.animateRing(sch, Nstep, dt, args, xlen / 2, k_lim, rp,save="Ring Off Centre")
tool.animateRing(sch, Nstep, dt, args, xlen / 2, k_lim, rp)