import numpy as np
import matplotlib.pyplot as plt


def oneRun(n, dt, x0, v0, a, b):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = x0
    for w in range(n - 1):
        t += dt
        tl.append(t)

        x += vol * dt
        xl.append(x)

        v += -a * xol * dt + b * np.sqrt(dt) * np.random.randn()
        # v += -a * x * dt
        vl.append(v)

        xol = x
        vol = v

    return tl, xl, vl


def manyRun(n_runs, n, dt, x0, v0, a, b, energy=False):
    runsx = np.zeros([n_runs, n])
    runsv = np.zeros([n_runs, n])
    runse = np.zeros([n_runs, n])
    for i in range(n_runs):
        # print(i)
        if i % 100 == 0:
            print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx[i] = np.real(xdat)
        runsv[i] = np.real(vdat)
        if energy:
            runse[i] = 0.5 * m * np.asarray(vdat) ** 2 + 0.5 * k * np.asarray(xdat) ** 2

    if energy:
        return tl, runsx, runsv, runse
    else:
        return tl, runsx, runsv


def averageRun(n_runs, n, dt, x0, v0, a, b, energy=False):
    runsx = np.zeros(n)
    runsv = np.zeros(n)
    for i in range(n_runs):
        # print(i)
        if i % 10 == 0:
            print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx += xdat / n_runs
        runsv += vdat / n_runs
    return tl, runsx, runsv


def expectedSolx(t, x0, v0, w):
    t = np.asarray(t)
    return x0 * np.cos(w * t) + v0 * np.sin(w * t) / w


def expectedSolv(t, x0, v0, w):
    t = np.asarray(t)
    return v0 * np.cos(w * t) - w * x0 * np.sin(w * t)


def expectedVar(t, sig, w):
    t = np.asarray(t)
    return sig ** 2 / (2 * w ** 2) * (t - np.sin(2 * w * t) / (2 * w))


def runsData(data):
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        if i % 10000 == 0:
            print(i)
        dslice = data[:, i]
        average[i] = np.mean(dslice)
        variance[i] = np.var(dslice)
    return average, variance


def energyRuns(run, n, dt, a, sig, x0=0):
    runs = np.zeros([run, n])
    for i in range(run):
        if i % 100 == 0:
            print(i)
        tl, xl, vl = oneRun(n, dt, x0, v0, a, sig)
        E = 0.5 * m * np.asarray(vl) ** 2 + 0.5 * k * np.asarray(xl) ** 2
        runs[i] = E
    return runs, tl


def energy(x, v, k, m):
    x = np.asarray(x)
    v = np.asarray(v)
    return 0.5 * m * v ** 2 + 0.5 * k * x ** 2


def expectedE(t, k, m, x0, v0, sig):
    t = np.asarray(t)
    return 0.5 * k * x0 ** 2 + 0.5 * m * v0 ** 2 + 0.5 * sig ** 2 * t/m


n = 50000
dt = 0.0001

m = 1
k = 10
w = np.sqrt(k / m)

sig = 1

x0 = 10
v0 = 0
print("t0",1/w)
print(dt)


### One Run
# tl, xl, vl = oneRun(n, dt, x0, v0, w ** 2, sig)
# plt.plot(tl, xl)
# plt.show()

# plt.plot(tl, vl)
# plt.show()

#### Many Run
# n_runs = 1000
# tl, runx, runv = manyRun(n_runs, n, dt, x0, v0, w ** 2, sig/m)
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for i in range(n_runs):
#     print(i)
#     ax1.plot(tl, runx[i])
#     ax2.plot(tl, runv[i])
#
# ax1.plot(tl, expectedSolx(tl, x0, v0, w), color="k", linestyle="--")
# ax2.plot(tl, expectedSolv(tl, x0, v0, w), color="k", label="Time Average", linestyle="--")
# fig.suptitle(f"Undamped Spring n={n_runs}")
# ax1.set_ylabel("x(t)")
# ax2.set_ylabel("v(t)")
# ax2.set_xlabel("t")
# ax2.legend(loc=8)
# plt.savefig("No_Damping")
# plt.show()

# x_av, x_var = runsData(runx)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle(f"Time Averaged n={n_runs}")
# ax1.plot(tl, x_av)
# ax1.plot(tl, expectedSolx(tl, x0, v0, w))
# ax2.plot(tl, x_var)
# ax2.plot(tl, expectedVar(tl, sig, w))
# ax1.set_ylabel("Expected value x")
# ax2.set_ylabel("Variance of x")
# ax2.set_xlabel("t")
# plt.savefig("No_Damping_Average")
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle(f"Difference n={n_runs}")
# ax1.plot(tl, abs(x_av - expectedSolx(tl, x0, v0, w)))
# ax2.plot(tl, abs(x_var - expectedVar(tl, sig, w)))
# ax1.set_ylabel("Expected value x")
# ax2.set_ylabel("Variance of x")
# ax2.set_xlabel("t")
# plt.savefig("No_Damping_Difference")
# plt.show()


#### Many Data
# n_run_list = [10, 100, 1000, 10000, 50000]
# # n_run_list = [10]
# for n_run in n_run_list:
#     print("Run Number", n_run)
#     t, x_act, v_act, e_act = manyRun(n_run, n, dt, x0, v0, w ** 2, sig,energy=True)
#     np.savez(f"Dat/NoDampRaw{n_run}x", x_act)
#     np.savez(f"Dat/NoDampRaw{n_run}v", v_act)
#     print("x Data")
#     av, var = runsData(x_act)
#     print("e Data")
#     ave, vare = runsData(e_act)
#     print("Saving")
#     np.savez(f"Dat/NoDamp{n_run}", (av, var, ave))
# np.savetxt("Dat/NoDampVariables", (n, dt, m, k, x0, v0))

##### Many Analysis
# np.savetxt("Dat/NoDampVariables.txt", (n, dt, m, k, x0, v0))
# n_run_list = [10, 100, 1000, 10000]
# n, dt, m, k, x0, v0 = np.loadtxt("Dat/NoDampVariables.txt")
# tl = np.asarray([i * dt for i in range(int(n))])
# w = np.sqrt(k / m)
# pos_expect = expectedSolx(tl, x0, v0, w)
# var_expect = expectedVar(tl, sig, w)
# e_expect = expectedE(tl,k,m,x0,v0,sig)
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for n_run in n_run_list:
#     dat = np.load(f"Dat/NoDamp{n_run}.npz")
#     average, variance, e_average = dat["arr_0"]
#     # average, variance, ave = np.load(f"Dat/NoDamp{n_run}.npz")
#     ax1.plot(tl, average, label=f"{n_run:.0e}")
#     ax2.plot(tl,average - pos_expect)
#
# ax1.plot(tl, pos_expect, color="k", linestyle="--", label="Expected")
# fig.suptitle(f"Average position")
# box = ax1.get_position()
# ax1.set_ylabel("Position")
# ax2.set_ylabel("Difference")
# ax2.set_xlabel("t")
# ax1.legend(loc=3)
# # plt.savefig(f"No_Damping_Position")
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for n_run in n_run_list:
#     dat = np.load(f"Dat/NoDamp{n_run}.npz")
#     average, variance, e_average = dat["arr_0"]
#     # average, variance, ave = np.load(f"Dat/NoDamp{n_run}.npz")
#     ax1.plot(tl, variance, label=f"{n_run:.0e}")
#     ax2.plot(tl, variance - var_expect)
#
# ax1.plot(tl,var_expect, color="k", linestyle="--", label="Expected")
# fig.suptitle(f"Variance")
# box = ax1.get_position()
# ax1.set_ylabel("Variance")
# ax2.set_ylabel("Difference")
# ax2.set_xlabel("t")
# ax1.legend(loc=2)
# # plt.savefig(f"No_Damping_Variance")
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for n_run in n_run_list:
#     dat = np.load(f"Dat/NoDamp{n_run}.npz")
#     average, variance, e_average = dat["arr_0"]
#     ax1.plot(tl, e_average, label=f"{n_run:.0e}")
#     ax2.plot(tl, e_average - e_expect)
# fig.suptitle(f"Energy")
# box = ax1.get_position()
# ax1.plot(tl, e_expect, color="k", linestyle="--", label="Expected")
# ax1.set_ylabel("Energy")
# ax2.set_ylabel("Difference")
# ax2.set_xlabel("t")
# ax1.legend(loc=2)
# # plt.savefig(f"No_Damping_Energy")
# plt.show()
#
# n_run_list = [1000, 10000]
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for n_run in n_run_list:
#     dat = np.load(f"Dat/NoDamp{n_run}.npz")
#     average, variance, e_average = dat["arr_0"]
#     ax1.plot(tl, abs(average - pos_expect), label=f"{n_run:.0e}")
#     ax2.plot(tl, abs(variance - var_expect))
#
# fig.suptitle(f"Difference of simulation and theory")
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax1.set_ylabel("Position")
# ax2.set_ylabel("Variance")
# ax2.set_xlabel("t")
# ax1.legend(bbox_to_anchor=(1.04, 1))
# # plt.savefig(f"No_Damping_Difference_Big")
# plt.show()

##### Energy
# n_run_list = [100, 1000, 10000]
# tl = np.asarray([i * dt for i in range(n)])
# E_theory = expectedE(tl, k, m, x0, v0, sig/m)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle("No Damping Energy")
# for n_run in n_run_list:
#     print("Run", n_run)
#     Edat, tl = energyRuns(n_run, n, dt, w ** 2, sig, x0=x0)
#     average, variance = runsData(Edat)
#     ax1.plot(tl, average, label=f"{n_run}")
#     ax2.plot(tl, abs(average - E_theory))
# ax1.plot(tl, expectedE(tl, k, m, x0, v0, sig), color="k", label="Theory")
# ax1.legend(loc=2)
# ax1.set_ylabel("Energy")
# ax2.set_ylabel("Energy Difference")
# ax2.set_xlabel("t")
# # plt.savefig("No_Damping_Energy_smaller")
# plt.show()

##### Initial Conditions
# dt_l = [0.01,0.001, 0.0001]
# time_tot = 2.5
# n_runs = 1000
# for dt in dt_l:
#     n = int(time_tot / dt)
#     tl, xl, vl = manyRun(n_runs, n, dt, x0, v0, w ** 2, sig)
#     average,var = runsData(xl)
#     plt.plot(tl,average - expectedSolx(tl, x0, v0, w),label=f"{dt}")
#
# plt.title(f"Different dt comparison, n={n_runs}")
# plt.xlabel("t")
# plt.ylabel("Difference of Sim and Theory")
# plt.legend()
# plt.savefig("No_Damping_dt")
# plt.show()
