import numpy as np
import matplotlib.pyplot as plt


def oneRun(n, dt, x0, v0, a, b):
    xl = [x0]
    vl = [g * x0 / (2 * m) + v0]
    t = 0
    tl = [0]
    x = x0
    v = g * x0 / (2 * m) + v0
    for w in range(n - 1):
        t += dt
        tl.append(t)

        x += v * dt
        xl.append(x)

        v += -a * x * dt + b * np.exp(g * t / (2 * m)) * np.sqrt(dt) * np.random.randn()
        vl.append(v)

    x_act = []
    v_act = []
    for i, t in enumerate(tl):
        x_act.append(xl[i] * np.exp(-g * t / (2 * m)))

        v_act.append(((vl[i] - (g * xl[i] / (2 * m))) * np.exp(-g * t / (2 * m))))
    return tl, x_act, v_act


def manyRun(n_runs, n, dt, x0, v0, a, b):
    runsx = np.zeros([n_runs, n])
    runsv = np.zeros([n_runs, n])
    for i in range(n_runs):
        # print(i)
        if i % 100 == 0:
            print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx[i] = np.real(xdat)
        runsv[i] = np.real(vdat)
    return tl, runsx, runsv


def runsData(data):
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        slice = data[:, i]
        average[i] = np.mean(slice)
        variance[i] = np.var(slice)
    return average, variance


def energy(x, v, k, m):
    x = np.asarray(x)
    v = np.asarray(v)
    return 0.5 * m * v ** 2 + 0.5 * k * x ** 2


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


def expectedVarx(t, g, m, sig, a, x0, v0):
    G = g / m
    w = np.sqrt(a)
    pref = sig ** 2 / (2 * G * (m ** 2) * a)
    co1 = G ** 2 / (G ** 2 + 4 * a)
    co2 = 2 * G * w / (G ** 2 + 4 * a)

    mat = pref * (1 + co1) + pref * np.exp(-G * t) * (1 - co1 * np.cos(2 * w * t) - co2 * np.sin(2 * w * t))
    initx = x0 ** 2 * (
            np.cos(w * t) ** 2 + (G / w) * np.sin(w * t) * np.cos(w * t) + (G ** 2 / 4 * a) * np.sin(w * t) ** 2)
    initxv = x0 * v0 * ((2 / w) * np.sin(w * t) * np.cos(w * t) + (G ** 2 / a) * np.sin(w * t) ** 2)
    initv = (v0 ** 2 / a) * np.sin(w * t) ** 2
    return mat + np.exp(-G * t) * (initx + initxv + initv)


def expectedVarx2(t, g, m, b, a):
    G = g / m
    w = np.sqrt(a)
    pref = b ** 2 / (2 * G * a)
    co1 = G ** 2 / (G ** 2 + 4 * a)
    co2 = 2 * G * w / (G ** 2 + 4 * a)

    # return pref * (1 + co1 - np.exp(-G * t) * (1 + co1 * np.cos(2 * w * t) - co2 * np.sin(2 * w * t)))
    return pref * (1 - co1 - np.exp(-G * t) * (1 - co1 * np.cos(2 * w * t) + co2 * np.sin(2 * w * t)))


n = 5000
dt = 0.001

g = 1
m = 1
k = 10

a = (g ** 2 / (2 * m ** 2)) + (k / m)
# a = -(g ** 2 / (4 * m ** 2)) + (k / m)

b = 2

x0 = 10
v0 = 0

############ One Run
# tl, x_act, v_act = oneRun(n, dt, x0, v0, a, b)
# x_expect = expectedSolx(tl, g, m, a, x0, v0)
# v_expect = expectedSolv(tl, g, m, a, x0, v0)
#
# plt.plot(tl, x_act)
# plt.plot(tl, x_expect, linestyle="--")
# plt.show()
#
# plt.plot(tl, v_act)
# plt.plot(tl, v_expect, linestyle="--")
# plt.show()

########### Many Run
# n_run = 10
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for i in range(n_run):
#     print(i)
#     tl, x_act, v_act = oneRun(n, dt, x0, v0, a, b)
#     ax1.plot(tl, x_act)
#     ax2.plot(tl, v_act)
#
# ax1.plot(tl, expectedSolx(tl, g, m, a, x0, v0), color="k", linestyle="--")
# ax2.plot(tl, expectedSolv(tl, g, m, a, x0, v0), color="k", label="Time Average", linestyle="--")
# fig.suptitle("Vibrating Spring")
# ax1.set_ylabel("x(t)")
# ax2.set_ylabel("v(t)")
# ax2.set_xlabel("t")
# ax2.legend(loc=8)
# plt.savefig("Stocastic_Spring")
# plt.show()

######### Energy Run
# n_run = 10
# ave = np.zeros(n)
# for i in range(n_run):
#     print(i)
#     tl, x_act, v_act = oneRun(n, dt, x0, v0, a, b)
#     e_tot = energy(x_act, v_act, k, m)
#     plt.plot(tl, e_tot)
#     for j in range(n):
#         ave[j] += e_tot[j]/n_run
#
# plt.show()
# plt.plot(tl, ave)
# plt.show()

########## Many run dat
# n_run = 10000
# t, x_act, v_act = manyRun(n_run, n, dt, x0, v0, a, b)
# ave = np.zeros(n)
# for i in range(n_run):
#     e_tot = energy(x_act[i], v_act[i], k, m)
#     for j in range(n):
#         ave[j] += e_tot[j] / n_run
#
# average, variance = runsData(x_act)
# np.savetxt("SpringData10000", (average, variance, ave))
#
# plt.plot(t, variance)
# plt.title("Averaged Variance")
# plt.xlabel("Time")
# plt.ylabel("Variance")
# plt.savefig("Spring_Variance_10000")
# plt.show()

########### Load dat
# n_load =10000
# average, variance, ave = np.loadtxt(f"Dat/BigData{n_load}.txt")
# n, dt, g, m, k, b = np.loadtxt("Dat/Variables.txt")
# a = g ** 2 / (2 * m ** 2) + k / m
# tl = np.asarray([i * dt for i in range(int(n))])
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.plot(tl, average)
# ax1.plot(tl, expectedSolx(tl, g, m, a, x0, v0), color="k", linestyle="--")
#
# ax2.plot(tl, variance, label="Simulated")
# ax2.plot(tl, expectedVarx2(tl,g,m,b,a), color="k", label="Time Average", linestyle="--")
# fig.suptitle(f"Average of Simulations n={n_load}")
# ax1.set_ylabel("x(t)")
# ax2.set_ylabel("v(t)")
# ax2.set_xlabel("t")
# ax2.legend(loc=8)
# plt.savefig(f"Stocastic_Spring_Averaged_{n_load}")
# plt.show()

##### Multiple run lengths
# n_run_list = [10, 100, 1000, 10000, 50000, 100000]
# for n_run in n_run_list:
#     t, x_act, v_act = manyRun(n_run, n, dt, x0, v0, a, b)
#     ave = np.zeros(n)
#     for i in range(n_run):
#         e_tot = energy(x_act[i], v_act[i], k, m)
#         for j in range(n):
#             ave[j] += e_tot[j] / n_run
#     av, var = runsData(x_act)
#     np.savetxt(f"Dat/BigData{n_run}.txt", (av, var, ave))
# np.savetxt("Dat/Variables.txt", (n, dt, g, m, k, b))

######## Multiple Run Analysis
n_run_list = [10, 100, 1000, 10000, 50000, 100000]
# n_run_list = [10000, 50000, 100000]
n, dt, g, m, k, b = np.loadtxt("Dat/Variables.txt")
tl = np.asarray([i * dt for i in range(int(n))])
a = g ** 2 / (2 * m ** 2) + k / m
pos_expect = expectedSolx(tl, g, m, a, x0, v0)
var_expect = expectedVarx2(tl, g, m, b, a)
# for n_run in n_run_list:
#     average, variance, ave = np.loadtxt(f"Dat/BigData{n_run}.txt")
#     plt.plot(tl, abs(variance - var_expect), label=f"{n_run:.0e}")
# plt.legend()
# plt.show()
#
# for n_run in n_run_list:
#     average, variance, ave = np.loadtxt(f"Dat/BigData{n_run}.txt")
#     plt.plot(tl, variance, label=f"{n_run}")
# plt.plot(tl, var_expect, label="Theory")
# plt.legend()
# plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
for n_run in n_run_list:
    average, variance, ave = np.loadtxt(f"Dat/BigData{n_run}.txt")
    ax1.plot(tl,abs(average-pos_expect), label=f"{n_run:.0e}")
    ax2.plot(tl, abs(variance - var_expect))

fig.suptitle(f"Difference of simulation and theory")
box = ax1.get_position()
ax1.set_position([box.x0,box.y0,box.width*0.8,box.height])
ax1.set_ylabel("Position")
ax2.set_ylabel("Variance")
ax2.set_xlabel("t")
ax1.legend(bbox_to_anchor=(1.04,1))
plt.savefig(f"Stocastic_Spring_Difference")
plt.show()