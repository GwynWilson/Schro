import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from PhD.Stochastic.Heller import Heller as hel
from scipy.integrate import simps

plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\FFmpeg\\bin\\ffmpeg.exe'


def hellerGaussian2D(x, y, vals, args):
    m, hbar, w = args
    xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
    return np.exp((1j / hbar) * (
            axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lam * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
            y - yt) + gt))


def psiSquare2D(psi):
    xlen, ylen = np.shape(psi)
    psi_s = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            psi_s[i, j] = np.real(psi[i, j] * np.conjugate(psi[i, j]))
    return psi_s


def hellerGaussian1D(x, y, vals, args):
    m, hbar, w = args
    xt, vt, at, gt = vals
    return np.exp((1j / hbar) * (at * (x - xt) ** 2 + m * vt * (x - xt) + gt))


def derivsdt(t, current, args, eta, dt):
    m, hbar, w1, w2 = args
    x = current[2] * dt
    y = current[3] * dt
    vx = -m * w1 ** 2 * current[0]
    vy = -m * w2 ** 2 * current[1]
    ax = -(2 / m) * current[4] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w1 ** 2
    ay = - (2 / m) * current[5] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w2 ** 2
    lam = -2 * ((current[4] / m) + (current[5] / m)) * current[6]
    gam = 1j * hbar * current[4] / m + 1j * hbar * current[5] / m + 0.5 * m * current[2] ** 2 + 0.5 * m * current[
        3] ** 2 - 0.5 * m * w1 ** 2 * current[0] ** 2 - 0.5 * m * w2 ** 2 * current[1] ** 2
    return x, y, vx, vy, ax, ay, lam, gam


n = 1000
dt = 0.001

w = 10
m = 1
hbar = 1
args = (m, hbar, w,w)
temp = 0.5 * m * w ** 2

x0 = 0
y0 = 0

vx0 = 0
vy0 = 0

lam = 0

sigx = np.sqrt(hbar / (2 * m * w))
sigy = np.sqrt(hbar / (2 * m * w))

ax0 = 1j * hbar / (4 * sigx ** 2)
ay0 = 1j * hbar / (4 * sigy ** 2)

sigt = np.sqrt(sigx ** 2 + sigy ** 2)

g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

############# Plot Demo
# x = np.linspace(-1,1,100)
# y = np.linspace(-1,1,100)
#
# X,Y = np.meshgrid(x,y)
#
# def foo(x,y,x0,y0,sig):
#     return np.exp(-(x-x0)**2/(2*sig**2) -(y-y0)**2/(2*sig**2))
#
# Z= foo(X,Y,0,0.5,0.1)
#
# fig, ax = plt.subplots()
# ax.pcolormesh(X,Y, Z)
# plt.show()
#
# fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
# ax.plot_surface(X,Y,Z)
# plt.show()


########### psi_demo
# x = np.linspace(-20, 20, 100)
# y = np.linspace(-20, 20, 100)
#
# X, Y = np.meshgrid(x, y)
#
# Z = hellerGaussian2D(X, Y, init, args)
# psi_s = np.real(Z*np.conjugate(Z))
#
# print(simps(simps(psi_s, x), y))
#
# # plt.imshow(psi_s)
# # plt.show()
#
# fig, ax = plt.subplots()
# a = ax.pcolormesh(x, y, np.asarray(psi_s))
# plt.colorbar(a)
# plt.show()

########### 2D animation demo
tl = np.asarray([i * dt for i in range(n)])
print(tl[-1])

vx = 0
xt = 5 * np.sin(100 * tl)
vxt = 500 * np.cos(100 * tl)

vals = init
vals[2] = vx

x = np.linspace(-20, 20, 500)
y = np.linspace(-20, 20, 500)

X, Y = np.meshgrid(x, y)

Z = hellerGaussian2D(X, Y, vals, args)
psi_s = np.real(Z * np.conjugate(Z))

fig, ax = plt.subplots()
# ax.set_xlim(-20, 20)
# ax.set_ylim(-20, 20)
# data = ax.pcolormesh(x, y, np.asarray(psi_s))
data = ax.imshow(psi_s, extent=[-20, 20, -20, 20])
cb = fig.colorbar(data)


def init():
    data.set_array(np.array(psi_s))
    return data


def animate(i):
    index = i % n
    # print(index)
    vals[0] = xt[index]
    vals[2] = vxt[index]

    vals[1] = 2 * xt[index]
    vals[3] = 2 * vxt[index]

    Z = hellerGaussian2D(X, Y, vals, args)
    psi_s = np.real(Z * np.conjugate(Z))

    data.set_array(np.array(psi_s))
    return data


ani = animation.FuncAnimation(fig, animate, interval=10, frames=int(n / 2), blit=False, init_func=init)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save('2D_Test.mp4', writer=writer)

plt.show()
