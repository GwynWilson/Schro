import numpy as np
import matplotlib.pyplot as plt


def ringPotential(x, y, x0, y0, m, w):
    r = np.sqrt(x ** 2 + y ** 2)
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    return 0.5 * m * w ** 2 * (r - r0) ** 2


def tiltedRing(x, y, r0, alpha, m, w):
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * m * w ** 2 * (r - r0) ** 2 + m * w ** 2 * r0 * alpha * x


matom = 1.44316072 * 10 ** -25
Natom = 10 ** 6
m = Natom * matom
hbar = 1.0545718 * 10 ** -34

w = 0.01
lim = 500 * 10 ** -6
# lim = 0.2

n_point = 500

x = np.linspace(-lim, lim, n_point)
y = np.linspace(-lim, lim, n_point)

r0 = 200 * 10 ** -6
x0 = 200 * 10 ** - 6
# x0 = 0.1/0.01
y0 = 0

vx0 = 0
vy0 = 100 * r0 * w

print("y velocity", vy0)

r0p = r0
x0p = r0
y0p = 0

print("packet radius x,r", x0p, np.sqrt(x0p ** 2 + y0p ** 2))

lam = 0

alpha = 0.4
# alpha = 1

sigx = 0.5 * 10 ** -4
sigy = 0.5 * 10 ** -4

ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)

g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

args = (m, hbar, w, r0)

xpacet = x0p + np.linspace(-2 * sigx, 2 * sigx, n_point)
ypacet = y0p + np.linspace(-2 * sigy, 2 * sigy, n_point)

xpacet = x
ypacet = y


#### Tilted Ring Plot
# X, Y = np.meshgrid(x, y)
# Z = tiltedRing(X, Y, r0, alpha, m, w)
#
# fig, ax = plt.subplots()
# data = ax.imshow(Z, extent=[-lim, lim, -lim, lim], cmap="gist_heat")
# cb = fig.colorbar(data)
# # theta = np.linspace(0, 2 * np.pi, 1000)
# # ax.plot(r0 * np.cos(theta), r0 * np.sin(theta), linestyle="--", color="w")
# plt.show()


############
potxnorm = ringPotential(xpacet,y0p,x0,y0,m,w)
potynorm = ringPotential(x0p,ypacet,x0,y0,m,w)

potx = tiltedRing(xpacet, y0p, r0, alpha, m, w)
poty = tiltedRing(x0p, ypacet, r0, alpha, m, w)


fig, ax = plt.subplots(2,2,sharey=True,figsize=(8,6))


ax[0,0].set_title("Tilted Ring")
ax[0,0].plot(xpacet / r0, potx / hbar, label="Potential")
ax[1,0].plot(ypacet / r0, poty / hbar, label="Potential")

ax[0,1].set_title("Normal Ring")
ax[0,1].plot(xpacet / r0, potxnorm / hbar, label="Potential")
ax[1,1].plot(ypacet / r0, potynorm / hbar, label="Potential")


ax[0,0].set_xlabel(r"$x/r_0$")
ax[1,0].set_xlabel(r"$y/r_0$")
ax[0,1].set_xlabel(r"$x/r_0$")
ax[1,1].set_xlabel(r"$y/r_0$")

ax[0,0].set_ylabel(r"$V(x,y=0)/\hbar$")
ax[1,0].set_ylabel(r"$V(x=r_0,y)\hbar$")

plt.savefig("Tilted Ring")
plt.show()
