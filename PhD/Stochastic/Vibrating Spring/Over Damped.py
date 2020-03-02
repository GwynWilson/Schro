import numpy as np
import matplotlib.pyplot as plt

from PhD.Stochastic.Ito_Process import itoProcess2, itoAverages, manyRuns


def a(x, t, args):
    return -x * args


def xAverage(t, x0, k, g):
    return x0 * np.exp(-k * t / g)


def xVariance(t, g, k, sig):
    return (sig ** 2 * g / (2 * k)) * (1 - np.exp(-2 * k * t / g))


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


n = 10000
dt = 0.001
t = np.asarray([i * dt for i in range(n)])

g = 1
k = 1
sig = 0.1
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
nrunlist = [5, 10, 100, 500]
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("Difference Of Simulation and Theory")
for i in nrunlist:
    runs = manyRuns(i, n, dt, a, sig, x0=x0, aargs=k / g)
    average, var = runsData(runs)
    ax1.plot(t,abs(average-xAverage(t,x0,k,g)))
    ax2.plot(t, abs(var - xVariance(t, g, k, sig)), label=f"{i}")
#
ax1.set_ylabel("Average Position")
ax2.set_ylabel("Average Variance")
ax2.set_xlabel("t")
ax2.legend()
plt.savefig("OverDifference")
plt.show()
