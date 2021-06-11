import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gauss(x, mu, sig):
    return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def plothist(dat, bins, label="", mu=None, lam=None):
    plt.hist(dat, bins=bins, density=True, label=label)
    if mu != None and lam != None:
        xrange = np.linspace(min(dat), max(dat), 2 * bins)
        plt.plot(xrange, gauss(xrange, 0, lam))

    return 0


def averageW(nruns, t, n):
    Ws_av = np.zeros(n)
    for i in range(nruns):
        dnk = gendnk(t, n)
        W = np.cumsum(dnk)
        Ws = W ** 2
        Ws_av += Ws / nruns
    return Ws_av


def gendnk(t, n):
    dt = t / n
    dnk = np.sqrt(dt) * np.random.randn(n)
    return dnk


bins = 50

################# Lambda Scaling
# t = 10
# n1 = 10000
# dt1 = t / n1
# dnk1 = np.sqrt(dt1) * np.random.randn(n1)
#
# n2 = 100000
# dt2 = t / n1
# dnk2 = np.sqrt(dt2) * np.random.randn(n2)
#
# plothist(dnk1,bins,label=f"{n1}")
# plothist(dnk2,bins,label=f"{n2}")
# plt.legend()
# plt.show()

################ Integration
# t = 10
# n = 10000
# dt = t / n
#
# dnk = gendnk(t,n)
# nt = np.cumsum(dnk)
#
# plt.plot(nt)
# plt.plot(dnk)
# plt.show()


############ White noise squared
# t = 10
# n1 = 10000
# n2 = 100000
#
# tl1 = [i * (t / n1) for i in range(n1)]
# tl2 = [i * (t / n2) for i in range(n2)]
# print("Times", tl1[-1], tl2[-1])
#
# dnk1 = gendnk(t, n1)
# dnk2 = gendnk(t, n2)
#
# dnks1 = [i ** 2 for i in dnk1]
# dnks2 = [i ** 2 for i in dnk2]
#
# nts1 = np.cumsum(dnks1)
# nts2 = np.cumsum(dnks2)
#
# plt.plot(tl1,nts1)
# plt.plot(tl2,nts2)
# plt.show()


########### W squared
# t = 10
# n1 = 10000
# n2 = 100000
#
# tl1 = [i * (t / n1) for i in range(n1)]
# tl2 = [i * (t / n2) for i in range(n2)]
# print("Times", tl1[-1], tl2[-1])
#
# dnk1 = gendnk(t, n1)
# dnk2 = gendnk(t, n2)
#
# W1 = np.cumsum(dnk1)
# W2 = np.cumsum(dnk2)
#
# Ws1 = [i ** 2 for i in W1]
# Ws2 = [i ** 2 for i in W2]
#
# plt.plot(tl1, Ws1, label=f"{n1}")
# plt.plot(tl2, Ws2, label=f"{n2}")
# plt.show()

################## Ws Averaged
t = 10
n = 10000
nruns = 10000

tl = [i * (t / n) for i in range(n)]
Ws = averageW(nruns, t, n)
plt.plot(tl, Ws)
plt.show()
