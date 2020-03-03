import numpy as np
import matplotlib.pyplot as plt

from PhD.Stochastic.Ito_Process import itoProcess2, itoAverages, manyRuns


def a(x, t, args):
    return -x * args


def xAverage(t, x0, k, g):
    return x0 * np.exp(-k * t / g)


def xVariance(t, g, k, sig):
    return (sig ** 2 * g / (2 * k)) * (1 - np.exp(-2 * k * t / g))


def eAverage(t, g, k, m, x0, sig):
    return (k / 2) * ((m * k / g ** 2) + 1) * (
            x0 ** 2 * np.exp(-2 * k * t / g) + (sig ** 2 * g / (2 * k)) * (1 - np.exp(-2 * k * t / g)))


def run(n, dt, a, b, x0):
    x = x0
    xl = [x0]
    for i in range(n - 1):
        x += a * x * dt + dt * np.random.randn()
        xl.append(x)
    return np.asarray(xl)


def runsData(data):
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        dslice = data[:, i]
        average[i] = np.mean(dslice)
        variance[i] = np.var(dslice)
    return average, variance


def energyRuns(run, n, dt, a, sig, x0=0, aargs=None):
    runs = np.zeros([run, n])
    for i in range(run):
        single = np.asarray(itoProcess2(n, dt, a, sig, x0=x0, aargs=aargs))
        xs = single**2
        vs = (-k*single/g)**2
        E = 0.5*m*vs+0.5*k*xs
        runs[i] = E
    return runs


def energy(x, v, k, m):
    x = np.asarray(x)
    v = np.asarray(v)
    return 0.5 * m * v ** 2 + 0.5 * k * x ** 2


n = 10000
dt = 0.001
t = np.asarray([i * dt for i in range(n)])

m = 1
g = 1
k = 1
sig = 1
x0 = 10

######### Single Run
# xdat = itoProcess2(n, dt, a, sig, x0=x0, aargs=k / g)
# plt.plot(t, xdat)
# plt.plot(t, xAverage(t, x0, k, g))
# plt.show()


######### Many Run
# nruns = 100
# runs = manyRuns(nruns, n, dt, a, sig, x0=x0, aargs=k / g)
# plt.figure()
# for i in range(nruns):
#     plt.plot(t, runs[i])
# plt.show()
#
# average, variance = runsData(runs)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle(f"Average and Variance n={nruns}")
# ax1.plot(t, average)
# ax1.plot(t, xAverage(t, x0, k, g), linestyle="--")
#
# ax2.plot(t, variance,label="Simulation")
# ax2.plot(t, xVariance(t, g, k, sig), linestyle="--",label="Theory")
#
# ax1.set_ylabel("Average Position")
# ax2.set_ylabel("Average Variance")
# ax2.set_xlabel("t")
# ax2.legend()
# plt.savefig(f"OverAverage{nruns}")
# plt.show()

######## Average with run length
# nrunlist = [5, 10, 100, 500]
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle("Difference Of Simulation and Theory")
# for i in nrunlist:
#     runs = manyRuns(i, n, dt, a, sig, x0=x0, aargs=k / g)
#     average, var = runsData(runs)
#     ax1.plot(t,abs(average-xAverage(t,x0,k,g)))
#     ax2.plot(t, abs(var - xVariance(t, g, k, sig)), label=f"{i}")
# #
# ax1.set_ylabel("Average Position")
# ax2.set_ylabel("Average Variance")
# ax2.set_xlabel("t")
# ax2.legend()
# plt.savefig("OverDifference")
# plt.show()


#### Energy
n_runs_list = [5,10, 100]
energy_expected = eAverage(t, g, k, m, x0, sig)
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("Overdamped Energy")
# ax1.set_yscale("log")
for nruns in n_runs_list:
    energy_runs = energyRuns(nruns, n, dt, a, sig, x0=x0, aargs=k / g)
    energy_av, var = runsData(energy_runs)
    ax1.plot(t, energy_av,label=f"{nruns}")
    ax2.plot(t, abs(energy_av - energy_expected))

ax1.plot(t, energy_expected, color="k", linestyle="--")
ax1.set_yscale("log")
ax1.legend()
ax1.set_ylabel("Energy")
ax2.set_ylabel("Energy Difference")
ax2.set_xlabel("t")
plt.savefig("OverEnergy")
plt.show()
