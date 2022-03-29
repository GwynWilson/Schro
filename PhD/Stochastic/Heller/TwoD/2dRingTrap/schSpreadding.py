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
sigx = 8
sigy = 8
# ax0 = 1j * hbar / (4 * sigx ** 2)
# ay0 = 1j * hbar / (4 * sigy ** 2)

ax0 = 1j * m * w / 2
ay0 = 1j * m * w / 2
lam = 0
g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]
args = (m, hbar, w, r0)

t = 0
T_tot = 4 * np.pi / w
Nstep = 500
dt = T_tot / Nstep

V = ring2D(X, Y, args)
psi = hellerGaussian2D(X, Y, init, args)

##### Animate Ring Spreadding
psis = np.real(psi * np.conjugate(psi))
# fig, ax = plt.subplots()
# ax.imshow(psis[::-1])
# plt.show()

sch = Schrodinger2D(x, y, psi, V, args)
# freeAnimate(1,3)
tool = PlotTools()
# tool.animateRing(sch, 1, dt, args, xlen / 2, k_lim, r0, save="Test")
tool.animateRing(sch, 1, dt, args, xlen / 2, k_lim, r0)

########## Changing with over time
t = 0
T_tot = 2 * np.pi / w
Nstep = 50
dt = T_tot / Nstep

sch = Schrodinger2D(x, y, psi, V, args)

theta = np.linspace(-np.pi, np.pi, N)
squared = np.real(sch.psi_x * np.conjugate(sch.psi_x))

psi_x = squared[:, xN]
psi_y = squared[int(N / 2), :]

psi_s = sch.getRing(theta, squared, r0)
plt.plot(x, psi_x, label="x")
plt.plot(y - r0, psi_y, label="y", linestyle="--")
plt.plot(theta * r0, psi_s, label="s", linestyle=":")
plt.legend()
plt.xlim(-50, 50)
plt.ylabel(r"$|\psi|^2$")
plt.xlabel("position")
plt.show()

t_l, s_l, wid_l, xwid_l, ywid_l = sch.ringExpectation(Nstep, dt, r0)
psi_f = sch.psi_x
psis_f = np.real(psi * np.conjugate(psi))

# tool = PlotTools()
# tool.compareInit(psi, psi_f, xlen / 2, k_lim)
#
# plt.plot(t_l,s_l)
# plt.show()

plt.title("Wave Packet Spreadding")
plt.plot(t_l, wid_l, label=r"$\sigma_s$")
plt.plot(t_l, xwid_l, label=r"$\sigma_x$")
plt.plot(t_l, ywid_l, label=r"$\sigma_y$")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Packet Width")
plt.show()
