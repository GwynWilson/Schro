from Schrodinger_Solver_2D import Schrodinger2D, PlotTools
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D, PlotTools2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fftpack import fftn, ifftn, fftshift


def gaussian(x, y, x0, y0, sigx, sigy):
    return (1 / np.sqrt(2 * np.pi * sigx * sigy) ** 2) * np.exp(-(x - x0) ** 2 / (2 * sigx ** 2)) * np.exp(
        -(y - y0) ** 2 / (2 * sigy ** 2))


def hellerGaussian2D(x, y, vals, args):
    m = args[0]
    hbar = args[1]
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def ring2D(x, y, args):
    m, hbar, w, r0 = args
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * m * w ** 2 * (r - r0) ** 2


def approxRing(x, y, args, xt, yt):
    m, hbar, w, r0 = args
    r = np.sqrt(x ** 2 + y ** 2)
    r0_t = np.sqrt(xt ** 2 + yt ** 2)
    v0 = 0.5 * m * w ** 2 * (r0_t - r0) ** 2
    vx = m * w ** 2 * xt * (1 - r0 / r0_t)
    vy = m * w ** 2 * yt * (1 - r0 / r0_t)
    vxx = 0.5 * m * w ** 2 * (1 + r0 * yt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
    vyy = 0.5 * m * w ** 2 * (1 + r0 * xt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
    vxy = m * w ** 2 * r0 * xt * yt / ((xt ** 2 + yt ** 2) ** (3 / 2))
    return v0 + vx * (x - xt) + vy * (y - yt) + vxx * (x - xt) ** 2 + vyy * (y - yt) ** 2 + vxy * (x - xt) * (y - yt)

def approxRingRot(x, y, args, xt, yt,theta):
    m, hbar, w, r0 = args
    r = np.sqrt(x ** 2 + y ** 2)
    r0_t = np.sqrt(xt ** 2 + yt ** 2)
    v0 = 0.5 * m * w ** 2 * (r0_t - r0) ** 2
    vxx = 0.5 * m * w ** 2 * (1 + r0 * yt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
    vyy = 0.5 * m * w ** 2 * (1 + r0 * xt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
    vxy = m * w ** 2 * r0 * xt * yt / ((xt ** 2 + yt ** 2) ** (3 / 2))

    vxxp = vxx*np.cos(theta)**2 + vxy*np.cos(theta)*np.sin(theta)+vyy*np.sin(theta)**2
    vyyp = vxx*np.sin(theta)**2 - vxy*np.cos(theta)*np.sin(theta)+vyy*np.cos(theta)**2

    return v0 + vxxp * x ** 2 + vyyp * y ** 2

########## Basic Transform
# N = 200
# xlen = 250
# dx = xlen / N
#
# x = np.asarray([i * dx for i in range(N)]) - xlen / 2
# y = np.asarray([i * dx for i in range(N)]) - xlen / 2
# X, Y = np.meshgrid(x, y)
#
# x0 = 0
# y0 = 0
# sigx = 12
# sigy = 12
#
# alpha=2
# beta =0.05
#
# gauss = gaussian(X+3*Y, Y+beta*X**2, x0, y0, sigx, sigy)
#
# fig, ax = plt.subplots()
# ax.imshow(gauss, extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()


############ Rotation
# N = 800
# xlen = 100
# dx = xlen / N
#
# x = np.asarray([i * dx for i in range(N)]) - xlen / 2
# y = np.asarray([i * dx for i in range(N)]) - xlen / 2
# X, Y = np.meshgrid(x, y)
#
# k_lim = np.pi / dx
# k_arr = -k_lim + (2 * k_lim / N) * np.arange(N)
#
# m = 1
# hbar = 1
#
# xN = int(3 * N / 4)
# w = 0.02  # set to stay within k range
# r0 = x[xN]
#
# x0 = r0
# y0 = 0
# vx0 = 0
# vy0 = 0
# sigx = 12
# sigy = 12
# # ax0 = 1j * hbar / (4 * sigx ** 2)
# # ay0 = 1j * hbar / (4 * sigy ** 2)
#
# ax0 = 1j * m * w / 2
# ay0 = 1j * m * w / 2
# lam = 1j * m * w / 2
# g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)
#
# init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
# args = (m, hbar, w, r0)
#
# X, Y = np.meshgrid(x, y)
#
# psi = hellerGaussian2D(X, Y, init, args)
# psis = np.real(np.conjugate(psi) * psi)
#
# fig, ax = plt.subplots()
# ax.imshow(psis[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()
#
# try:
#     theta = 0.5 * np.arctan(lam / (ax0 - ay0))
# except ZeroDivisionError:
#     theta = np.pi/4 # Breaks when ax0=ay0 but the limit for theta is pi/4
# thetar = np.real(theta)
# ct = np.cos(theta)
# st = np.sin(theta)
#
# ax0p = ax0 * ct ** 2 + lam * ct * st + ay0 * st ** 2
# lamp = 0
# ay0p = ax0 * st ** 2 - lam * ct * st + ay0 * ct ** 2
#
# initp = [x0, y0, vx0, vy0, ax0p, ay0p, lamp, g0]
#
# psir = hellerGaussian2D(X,Y,initp,args)
# psirs = np.real(np.conjugate(psir) * psir)
# fig, ax = plt.subplots()
# ax.imshow(psirs[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()


#################### Transforming potential
N = 200
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
w = 0.002  # set to stay within k range
r0 = x[xN]

angle = np.pi/4
x0 = r0 *np.cos(angle)
y0 = r0 *np.sin(angle)

# x0 = r0
# y0 = 0

vx0 = 0
vy0 = 0
sigx = 8
sigy = 8
# ax0 = 1j * hbar / (4 * sigx ** 2)
# ay0 = 1j * hbar / (4 * sigy ** 2)

ax0 = 1j * m * w / 2
ay0 = 1j * m * w / 2
lam = 1j * m * w / 2
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar, w, r0)

psi = hellerGaussian2D(X, Y, init, args)
psis = np.real(np.conjugate(psi) * psi)

V = ring2D(X, Y, args)
V_app = approxRing(X, Y, args, x0, y0)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(psis[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# ax[1].imshow(V[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(V_app[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# ax[1].imshow(V[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()

# Displacement of co-ordinates
# x = x + x0
# y = y + y0
# X, Y = np.meshgrid(x, y)
# psi = hellerGaussian2D(X, Y, init, args)
# psis = np.real(np.conjugate(psi) * psi)
#
# V = ring2D(X, Y, args)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(psis[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# ax[1].imshow(V[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
# plt.show()

# Rotation
xt = x0
yt = y0
vxx = 0.5 * m * w ** 2 * (1 + r0 * yt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
vyy = 0.5 * m * w ** 2 * (1 + r0 * xt ** 2 / ((xt ** 2 + yt ** 2) ** (3 / 2)))
vxy = m * w ** 2 * r0 * xt * yt / ((xt ** 2 + yt ** 2) ** (3 / 2))

try:
    thetap = 0.5 * np.arctan(vxy / (vxx - vyy))
except ZeroDivisionError:
    thetap = np.pi / 4  # Breaks when ax0=ay0 but the limit for theta is pi/4

try:
    theta = 0.5 * np.arctan(lam / (ax0 - ay0))
except ZeroDivisionError:
    theta = np.pi / 4  # Breaks when ax0=ay0 but the limit for theta is pi/4

print(thetap,theta)
theta = thetap

x = np.asarray([i * dx for i in range(N)]) - xlen / 2
y = np.asarray([i * dx for i in range(N)]) - xlen / 2
X, Y = np.meshgrid(x, y)

Xp = X * np.cos(theta) - Y * np.sin(theta) + x0
Yp = X * np.sin(theta) + Y * np.cos(theta) + y0

psi = hellerGaussian2D(Xp, Yp, init, args)
psis = np.real(np.conjugate(psi) * psi)

V = ring2D(Xp, Yp, args)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(psis[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
ax[1].imshow(V[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(approxRingRot(X,Y,args,x0,y0,theta)[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
ax[1].imshow(approxRing(Xp, Yp, args, x0, y0)[::-1], extent=[-xlen / 2, xlen / 2, -xlen / 2, xlen / 2])
plt.show()