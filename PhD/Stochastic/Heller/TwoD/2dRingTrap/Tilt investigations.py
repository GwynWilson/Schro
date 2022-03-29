from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D, PlotTools2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fftpack import fftn, ifftn, fftshift
import scipy.misc


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


def freeAnimate(Nstep, dt):
    sch = Schrodinger2D(x, y, psi, V, args)
    tool = PlotTools()
    tool.animation2D(sch, Nstep, dt, args, xlen / 2, k_lim)


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

x0 = r0
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

t = 0
T_tot = np.pi / w
Nstep = 500
dt = T_tot / Nstep

V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)

########
x0 = r0
vy0 = 1

x0 = 0.5 * (r0 + np.sqrt(r0 ** 2 + 4 * (vy0 ** 2 / w ** 2)))

# x0 = 70

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)

#### Initial
# psis = np.real(psi * np.conjugate(psi))
# fig, ax = plt.subplots()
# ax.imshow(psis[::-1])
# plt.show()

#######Compare Inital and Final
theta = np.linspace(-np.pi, np.pi, N)
thetax = x0 * np.cos(theta)
thetay = x0 * np.sin(theta)
thetadup = (thetax, thetay)

# sch = Schrodinger2D(x, y, psi, V, args)
# sch.evolvet(Nstep, dt)
# np.savez_compressed("psi_tilt",psi=sch.psi_x,time=sch.t)

loaded = np.load("psi_tilt.npz")
psif = loaded["psi"]

tool = PlotTools()
tool.compareInit(psi, psif, xlen / 2, k_lim, other=thetadup)


######## Tilt investigation
loaded = np.load("psi_tilt.npz")
psif = loaded["psi"]
psifs = np.real(psif * np.conjugate(psif))

X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(k_arr, k_arr)

x_expect = simps(simps(X * psifs, x), y)
y_expect = simps(simps(Y * psifs, y), x)

xs_expect = simps(simps(X ** 2 * psifs, x), y)
ys_expect = simps(simps(Y ** 2 * psifs, y), x)
xy_expect = simps(simps(X * Y * psifs, x), y)
yx_expect = simps(simps(Y * X * psifs, y), x)

cov_x = xs_expect - x_expect ** 2
cov_y = ys_expect - y_expect ** 2
cov_xy = xy_expect - (x_expect * y_expect)

matrix = np.matrix([[cov_x, cov_xy], [cov_xy, cov_y]])
eigenval, eigenvecs = np.linalg.eig(matrix)
print(eigenval, eigenvecs)
eigenvecs = np.asarray(eigenvecs)

# fig,ax= plt.subplots()
# ax.quiver(0,0,eigenval[0]*eigenvecs[0][0],eigenval[0]*eigenvecs[0][1],units="xy",scale=1)
# ax.quiver(0,0,eigenval[1]*eigenvecs[1][0],eigenval[1]*eigenvecs[1][1],units="xy",scale=1)
# plt.xlim(-75,75)
# plt.ylim(-75,75)
# plt.show()
# print(matrix)

thet = 0.5 * np.arctan(2 * cov_xy / (cov_x - cov_y))

xold = x - x_expect
yold = np.asarray([0 for i in range(len(xold))])

xnew = xold * np.cos(thet) - yold * np.sin(thet)
ynew = xold * np.sin(thet) + yold * np.cos(thet)

yold2 = y - y_expect
xold2 = np.asarray([0 for i in range(N)])

xnew2 = xold2 * np.cos(thet) - yold2 * np.sin(thet)
ynew2 = xold2 * np.sin(thet) + yold2 * np.cos(thet)

# print(xs_expect-x_expect**2,xy_expect-x_expect*y_expect,ys_expect-y_expect**2,yx_expect-x_expect*y_expect)


# print(x_expect, y_expect)
psik = fftshift(fftn(psif))
psik *= np.exp(1j * x_expect * KX)
psik *= np.exp(1j * y_expect * KY)

psif = ifftn(fftshift(psik))
psifs = np.real(psif * np.conjugate(psif))

plotlim = 4 * sigx

fig, ax = plt.subplots()
fig.suptitle("Tilt Zoom")
ax.imshow(psifs[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
ax.plot(thetax - x_expect, thetay - y_expect, linestyle="--", color="w")
ax.plot(xnew, ynew, linestyle="--", color="k", alpha=0.5)
ax.plot(xnew2, ynew2, linestyle="--", color="k", alpha=0.5)

ax.set_xlim(-plotlim, plotlim)
ax.set_ylim(-plotlim, plotlim)
# ax.text(-0.95 * plotlim, 0.9 * plotlim, 'test', fontsize=10, color="w")
plt.show()

######### Free animation
# freeAnimate(1,3)
# tool = PlotTools()
# tool.animateRing(sch, 1, dt, args, xlen / 2, k_lim, r0, save="Test")
# tool.animateRing(sch, 1, dt, args, xlen / 2, k_lim, r0)
