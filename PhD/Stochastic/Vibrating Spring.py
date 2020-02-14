import numpy as np
import matplotlib.pyplot as plt
from Ito_Process import weinerProcess


def oneRun(n, dt, x0, v0, a, b):
    xl = [x0]
    vl = [g * x0 / (2 * m) + v0]
    t = 0
    tl = [0]
    x = x0
    v = g * x0 / (2 * m) + v0
    wt = weinerProcess(n, dt)
    for w in wt:
        t += dt
        tl.append(t)

        x += v * dt
        xl.append(x)

        v += -a * x * dt + b * np.exp(g * t / (2 * m)) * w
        vl.append(v)

    x_act = []
    v_act = []
    for i, t in enumerate(tl):
        x_act.append(xl[i] * np.exp(-g * t / (2 * m)))
        # v_act.append(((vl[i]) * np.exp(-g * t / (2 * m))) - g * xl[i] / (2 * m))

        v_act.append(((vl[i] - (g * xl[i] / (2 * m))) * np.exp(-g * t / (2 * m))))
    return tl, x_act, v_act


def expectedSolx(t, g, m, a, x0, v0):
    t = np.asarray(t)
    w = np.sqrt(a)
    return np.exp(-g * t / (2 * m)) * (x0 * np.cos(w * t) + (g * x0 / (2 * m) + v0) * np.sin(w * t) / w)


def expectedSolv(t, g, m, a, x0, v0):
    t = np.asarray(t)
    w = np.sqrt(a)
    return np.exp(-g * t / (2 * m)) * (-x0 * w * np.sin(w * t) + g * x0 * np.cos(w * t) / (2 * m) + v0 * np.cos(w * t)) \
           - g * expectedSolx(t, g, m, a, x0, v0) / (2 * m)


def expectedSolv2(t, g, m, a, x0, v0):
    t = np.asarray(t)
    w = np.sqrt(a)
    v20 = g * x0 / (2 * m) + v0
    return -x0 * np.sin(w * t) / w + v20 * np.cos(w * t)


n = 50000
dt = 0.0001

g = 0
m = 1
k = 10

a = (g ** 2 / (2 * m ** 2)) + (k / m)
a = -(g ** 2 / (4 * m ** 2)) + (k / m)
print("a", a)
print("Period", 2 * np.pi / np.sqrt(a))

b = 0.05

x0 = 10
v0 = 0

print("maxv", x0 * (1 / np.sqrt(a) - g / (2 * m)))

############ One Run
# tl, x_act, v_act = oneRun(n, dt, x0, v0, a, b)
# x_expect = expectedSolx(tl, g, m, a, x0, v0)
# v_expect = expectedSolv(tl, g, m, a, x0, v0)
# # v_expect = expectedSolv2(tl, g, m, a, x0, v0)
#
# plt.plot(tl, x_act)
# plt.plot(tl, x_expect, linestyle="--")
# plt.show()
#
# plt.plot(tl, v_act)
# plt.plot(tl, v_expect, linestyle="--")
# plt.show()

########### Many Run
n_run = 10
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
for i in range(n_run):
    print(i)
    tl, x_act, v_act = oneRun(n, dt, x0, v0, a, b)
    ax1.plot(tl, x_act)

    ax2.plot(tl, v_act)

ax1.plot(tl, expectedSolx(tl, g, m, a, x0, v0), color="k")
ax2.plot(tl, expectedSolv(tl, g, m, a, x0, v0), color="k", label="Time Average")
fig.suptitle("Vibrating Spring")
ax1.set_ylabel("x(t)")
ax2.set_ylabel("v(t)")
ax2.set_xlabel("t")
ax2.legend(loc=8)
# plt.savefig("Stocastic_Spring_3")
plt.show()
