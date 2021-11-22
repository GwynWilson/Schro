import numpy as np
import matplotlib.pyplot as plt
from PhD.Stochastic.Heller.TwoD.Heller2D import Heller2D, PlotTools2D


def hellerGaussian2D(x, y, vals, args):
    m, hbar = args
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lam * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def derivsDt(t, current, args, eta, dt):
    m, hbar = args
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = current
    xn = vxt * dt
    yn = vyt * dt
    vxn = 0
    vyn = 0
    axn = (-2 * axt ** 2 / m - 0.5 * lamt ** 2 / m)*dt
    ayn = (-2 * ayt ** 2 / m - 0.5 * lamt ** 2 / m)*dt
    lamn = (-2 * (axt + ayt) * lamt / m)*dt
    gn = (1j * hbar * axt / m + 1j * hbar * ayt / m + 0.5 * m * vxt ** 2 + 0.5 * m * vyt ** 2)*dt
    return xn, yn, vxn, vyn, axn, ayn, lamn, gn


matom = 1.44316072 * 10 ** -25
Natom = 10 ** 6
m = Natom * matom
hbar = 1.0545718 * 10 ** -34
w = 0.1
lim = 300 * 10 ** -6

n_point = 1000
x = np.linspace(-lim, lim, n_point)
y = np.linspace(-lim, lim, n_point)

x0 = 0
y0 = 0

vx0 = 0
vy0 = 0

lam = 0

sigx = 10 ** -4
sigy = 10 ** -4

ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)

g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

args = (m, hbar)

#######Initial Plot
# X, Y = np.meshgrid(x, y)
# psi = hellerGaussian2D(X, Y, init, args=args)
# Z = np.real(psi*np.conjugate(psi))
#
# fig, ax = plt.subplots()
# data = ax.imshow(Z, extent=[-lim, lim, -lim, lim], cmap="gist_heat")
# cb = fig.colorbar(data)
# plt.show()


########Simulation
# n = 1000
# dt = 0.001

tf = 2 * m * sigx * sigy / hbar
n = 1000
dt = tf / n

Hel = Heller2D(n, dt, init, derivsDt)
tl, var_arr = Hel.rk4(args)
# print(var_arr[:,0])

Plot = PlotTools2D(tl, var_arr, lim, lim, n_point)
# Plot.animation2D(args)
# Plot.plot2D(var_arr[:, 0], args)
# Plot.plot2D(var_arr[:, -1], args)

Plot.animation2D(args,save="Spread_3")