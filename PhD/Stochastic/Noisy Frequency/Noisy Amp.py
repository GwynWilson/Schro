import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.integrate import simps


def detectFolder():
    import os
    cwd = os.getcwd()
    folder = "Dat"
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except OSError:
            print("whoops")


def genNoise(n, dt):
    """
    Will generate array of length 2n of gaussian white noise
    :param n:
    :return:
    """
    return np.sqrt(dt) * np.random.randn(2 * n)


def f(t, x, v):
    return v


def g(t, x, v, a, eta, b):
    return -a * b * x * eta


def rk4StocasticPhase(variables, noise=None):
    n, dt, x0, v0, w, sig, m = variables
    a = w ** 2
    b = sig / m
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    if not isinstance(noise, np.ndarray):
        noise = genNoise(n, dt)
    for i in range(n - 1):
        k0 = dt * f(t, x, v)
        l0 = dt * g(t, x, v, a, noise[2 * i], b)

        k1 = dt * f(t + dt / 2, x + k0 / 2, v + l0 / 2)
        l1 = dt * g(t + dt / 2, x + k0 / 2, v + l0 / 2, a, noise[2 * i], b)

        k2 = dt * f(t + dt / 2, x + k1 / 2, v + l1 / 2)
        l2 = dt * g(t + dt / 2, x + k1 / 2, v + l1 / 2, a, noise[2 * i], b)

        k3 = dt * f(t + dt, x + k2, v + l2)
        l3 = dt * g(t + dt, x + k2, v + l2, a, noise[2 * i + 2], b)

        x += (k0 + 2 * k1 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6
        t += dt

        xl.append(x)
        vl.append(v)
        tl.append(t)

    return tl, np.asarray(xl), np.asarray(vl)


def expect(variables, noise, tl):
    tl = np.asarray(tl)
    n, dt, x0, v0, w, sig, m = variables
    noise = noise[::2]
    W = np.cumsum(noise,dtype="complex")
    print((sig * w ** 2 / 2) * (sig * w ** 2 * tl + 2 * W))
    wp = np.sqrt((sig * w ** 2 / 2) * (sig * w ** 2 * tl + 2 * W))
    return x0 * np.cos(wp * np.sqrt(tl))


d = 0.004
v = 0.01
t = d / v
print("Final time", t)
n = 500
dt = t / n

m = 1.44 * 10 ** (-25)
w = 6.1 * 10 ** 2
k = w ** 2 * m

print("t0", w ** (-1))
hbar = 1.0545718 * 10 ** -34

sig = m / 100

x0 = 0.0001
v0 = 0

var_list = [n, dt, x0, v0, w, sig, m]

noise = genNoise(n, dt)
tl, xl, vl = rk4StocasticPhase(var_list, noise=noise)

plt.plot(tl, xl)
plt.plot(tl, expect(var_list, noise, tl))
plt.show()
