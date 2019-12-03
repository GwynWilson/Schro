from Ito_Process import *
import matplotlib.pyplot as plt
import numpy as np


def a(x, t, args):
    return -x * args


def b(x, t, args):
    return np.sqrt(args)


def expect(x0,k, t):
    return x0*np.exp(-k * t)


n = 1000
dt = 0.0001
t_list = np.array([i * dt for i in range(n)])
k = 1
D = 1
x0=0

# nruns = 100
# tot_runs = manyRuns(nruns, n, dt, a, b,x0=x0, aargs=(k), bargs=(D))
#
# plotRuns(tot_runs)
#
#
# plt.plot(t_list,itoAverages(tot_runs))
# plt.plot(t_list, expect(x0,k, t_list))
# plt.show()

runlist = [10,100,1000]
for i in runlist:
    tot_runs = manyRuns(i,n,dt,a, b,x0=x0, aargs=(k), bargs=(D))
    plt.plot(t_list,itoAverages(tot_runs),label=f"{i}")
plt.legend()
plt.show()