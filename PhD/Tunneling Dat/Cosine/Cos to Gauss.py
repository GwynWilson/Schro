from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameters_Realistic import *
from scipy.optimize import curve_fit
from scipy.integrate import simps


def cosToGauss(x, wid, sig):
    if x > -wid and x < wid:
        k = np.pi / wid
        exp_wid = 1 / k
        # exp_wid = sig
        # return (1 + np.cos(k * x)) / 2 * np.exp(-x ** 2 / sig ** 2)
        return (1 + np.cos(k * x)) / 2 * np.exp(-x ** 2 / exp_wid ** 2)
    else:
        return 0


def tanh(x, V0, a, w):
    return 0.5 * V0 * (np.tanh((x + a) / w) - np.tanh((x - a) / w)) / (np.tanh(a / w))


def gauss(x, A, sig):
    return A * np.exp(-x ** 2 /(sig ** 2))


def cos(x, L, A):
    x = np.asarray(x)
    return 0.5 * (1 + np.cos(np.pi * x / L))


def cos2(x, L, A):
    x = np.asarray(x)
    bar = []
    for i in x:
        if i > -L and i < L:
            bar.append(A* (1 + np.cos(np.pi * i / L))/2)
        else:
            bar.append(0)
    return np.asarray(bar)


def gwynCos(x, sig, par):
    l=sig * (1.5*(np.pi**(5/2)/(np.pi**2-6)))**(1/3)
    # l=(sig**2/0.130691)**(1/3)
    # return (par * gauss(x, 1, sig) + (1 - par) * cos2(x, 2*sig, 1)) / 2
    return (par * gauss(x, 2, sig) + (1 - par) * cos2(x, l, 2)) / 2

def retriveBar(x, L, bar):
    dat = []
    xs = []
    for n, v in enumerate(x):
        if v > -L and v < L:
            xs.append(v)
            dat.append(bar[n])
    return xs, dat


############Cos
# cb = [cosToGauss(i, L, sig) for i in x]
# cb2 = [cosToGauss(i, 2 * L, sig) for i in x]
# cb3 = [cosToGauss(i, 0.5 * L, sig) for i in x]
#
# plt.plot(x * 10 ** 6, cb3, label=f"{0.5*L}")
# plt.plot(x * 10 ** 6, cb, label=f"{L}")
# plt.plot(x * 10 ** 6, cb2, label=f"{2*L}")
#
# plt.title("Cosine Barrier for varying L")
# plt.xlabel("x(micrometers)")
# plt.ylabel("V(x)")
# plt.legend()
# plt.xlim(-2 * L * 10 ** 6, 2 * L * 10 ** 6)
# plt.savefig("Cosine to Gauss")
# plt.show()

######## Curve Fit Cos
# wid = 15*L
#
# cb4 = [cosToGauss(i, wid, sig) for i in x]
# popt, covt = curve_fit(gauss, x, cb4, (1, sig))
#
# xc,bar_dat = retriveBar(x,wid,cb4)
#
# popt2, covt2 = curve_fit(cos, xc, bar_dat, (wid, 1))
#
# plt.plot(x, cb4)
# plt.plot(x, gauss(x, popt[0], popt[1]))
# plt.plot(xc,cos(xc,popt2[0],popt2[1]))
# # plt.plot(xc,cos(xc,wid,1))
# plt.show()

#############tanh test
# w=10**(-10)
# plt.plot(x,tanh(x,bar_amp,L/2,w)/scale)
# plt.show()


########Gwyn
# plt.plot(x, bar_amp*gwynCos(x, 4*sig, 1))
# plt.plot(x, bar_amp*gwynCos(x, 4*sig, 0.5))
# plt.plot(x, bar_amp*gwynCos(x, 4*sig, 0))
# plt.show()

#######Gwyn int
# cosbar = gwynCos(x, sig, 0)
# gbar = gwynCos(x, sig, 1)
# xscosbar = np.zeros(len(x))
# xsgbar = np.zeros(len(x))
# for i,v in enumerate(x):
#     xscosbar[i] = cosbar[i]
#     xsgbar[i] = gbar[i]*v**2
#
# print(simps(xscosbar,x))
# print(simps(xsgbar,x))
# print(sig**3 * np.sqrt(np.pi)/2)
# print(sig)

