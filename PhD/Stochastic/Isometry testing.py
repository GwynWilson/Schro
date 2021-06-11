import numpy as np
import matplotlib.pyplot as plt


def genNoise(n, dt):
    """
    Will generate array of length 2n of gaussian white noise
    :param n:
    :return:
    """
    return np.sqrt(dt) * np.random.randn(2 * n)


def euler(n, x0, dt, b, noise=None):
    x = x0
    xl = [x0]
    t = 0
    tl = [0]
    if not isinstance(noise, np.ndarray):
        noise = genNoise(n, dt)
    for i in range(n - 1):
        x += b(t) * noise[i]
        t += dt
        tl.append(t)
        xl.append(x)
    return tl, xl


def nons(n, x0, dt, b):
    x = x0
    xl = [x0]
    t = 0
    tl = [0]
    for i in range(n - 1):
        x += b(t) ** 2 * dt
        t += dt
        tl.append(t)
        xl.append(x)
    return tl, xl


def manyRun(n_run, n, dt, x0, b):
    x_sum = np.zeros(n)
    x_sums = np.zeros(n)
    conv = []
    for i in range(n_run):
        tl, xl = euler(n, x0, dt, b)
        x_sum += xl
        x_sums += np.asarray(xl) ** 2
        if i % 100 == 0 and i != 0:
            conv.append(x_sum / (i * 100))
    return tl, x_sum, x_sums, conv


def dualrun(n_run, n, dt, x0, f, g):
    x_sums = np.zeros(n)

    for i in range(n_run):
        noise = genNoise(n, dt)
        tl1, xl1 = euler(n, x0, dt, f, noise=noise)
        tl2, xl2 = euler(n, x0, dt, g, noise=noise)
        x_sums += (np.asarray(xl1) ** 2) * (np.asarray(xl2) ** 2)
        if i % 1000 == 0:
            print(i)

    return tl1, x_sums


def isomRun(n_run, n, dt, x0, f):
    x_sums_same = np.zeros(n)
    x_sums_diff = np.zeros(n)

    for i in range(n_run):
        noise = genNoise(n, dt)
        tl1, xl1 = euler(n, x0, dt, f, noise=noise)
        tl2, xl2 = euler(n, x0, dt, f, noise=noise)
        x_sums_same += np.asarray(xl1) ** 2
        x_sums_diff += np.asarray(xl1) * np.asarray(xl2)
        if i % 1000 == 0:
            print(i)

    return tl1, x_sums_same, x_sums_diff


def quadrun(n_run, n, dt, x0, f, g):
    x_sums = np.zeros(n)

    for i in range(n_run):
        tl1, xl1 = euler(n, x0, dt, f)
        tl2, xl2 = euler(n, x0, dt, f)
        tl3, xl3 = euler(n, x0, dt, g)
        tl4, xl4 = euler(n, x0, dt, g)
        x_sums += np.asarray(xl1) * np.asarray(xl2) * np.asarray(xl3) * np.asarray(xl4)
        if i % 1000 == 0:
            print(i)

    return tl1, x_sums


def f(t):
    return np.cos(t)


def g(t):
    return np.sin(t)


def integ(T):
    T = np.asarray(T)
    return (-1 + 8 * T ** 2 + np.cos(4 * T)) / 32


def integAdd(T):
    T = np.asarray(T)
    return np.sin(T) ** 4 / 2


def cosInt(t):
    T = np.asarray(t)
    return 0.5 * (T + np.cos(T) * np.sin(T))


n = 5000
dt = 0.001
x0 = 0
b = 1

n_runs = 50000
# n_runs = 10000

w = genNoise(n, dt)

###### Isometry test
# tl, xl, xsl, conv = manyRun(n_runs, n, dt, x0, f)
# tl2, xl2 = nons(n, x0, dt, f)
#
# plt.plot(tl, xsl/n_runs)
# plt.plot(tl2, xl2)
# plt.show()


##### Dual isometry
# tl, simmeth = dualrun(n_runs, n, dt, x0, f, g)
# np.savez_compressed(f"Isometry_Test_{n_runs}_same", tl=tl, simmeth=simmeth)

dat = np.load(f"Isometry_Test_{n_runs}_same.npz")
tl = dat["tl"]
simmeth = dat["simmeth"]

# plt.plot(tl, simmeth/n_runs)
# plt.show()

# tc,xc = nons(n,x0,dt,f)
# ts,xs = nons(n,x0,dt,g)
tl = [i * dt for i in range(n)]
# calcmeth = np.asarray(integ(tl))
calcweird = np.asarray(integ(tl) + integAdd(tl))

# plt.title("Higher Order Isommetry relation")
# plt.plot(tl, simmeth / n_runs, label="Simulation")
# plt.plot(tl, calcmeth,linestyle="--", label="Analytic")
# plt.legend()
# plt.savefig("Isommetry_Relation")
# plt.show()

plt.title("2nd Order Isommetry relation")
plt.plot(tl, simmeth / n_runs, label="Simulation")
# plt.plot(tl, calcmeth, label="Analytic")
plt.plot(tl, calcweird,linestyle="--", label="Analytic")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Integral (Unitless)")
plt.savefig("Isommetry Relation 2 ext")
plt.show()


################# Isometry Basic
# tl, sam, diff = isomRun(n_runs, n, dt, x0, f)
# np.savez_compressed(f"Isometry_Test_{n_runs}_basic", tl=tl, sam=sam, diff=diff)
#
# dat = np.load(f"Isometry_Test_{n_runs}_basic.npz")
# tl = dat["tl"]
# sam = dat["sam"]
# diff = dat["diff"]
#
# plt.plot(tl, sam / n_runs, label="Same")
# plt.plot(tl, diff / n_runs, label="Different")
# plt.plot(tl, cosInt(tl), linestyle="--", label="Analytic")
# plt.savefig("Basic Isometry")
# plt.show()
