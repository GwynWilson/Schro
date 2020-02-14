from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import curve_fit


# from Input_Parameters_Realistic import *


def gauss(x, mu=0, sig=1):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def gaussfit(x, mu, sig, A):
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def barrier(x, w):
    return np.tanh((gauss(x)) / w) / np.tanh((gauss(0)) / w)


def barrier2(x, w):
    raw = np.tanh(np.exp(-x ** 2) / w)
    return raw / simps(raw, x)


def barrier3(x, w, L):
    sig = np.sqrt(L ** 2 / (8 * np.log(2 / w)))
    raw = np.tanh((gauss(x, sig=sig)) / w) / np.tanh((gauss(0, sig=sig)) / w)
    return raw


def barrier4(x, w, L):
    raw = np.tanh((2 / w) ** (-4 * (x ** 2) / L ** 2) / w) / np.tanh(1 / w)
    return raw


def barrier5(x, w, L):
    if np.log(2 / w) > 1:
        raw = np.tanh((2 / w) ** (-4 * (x ** 2) / L ** 2) / w) / np.tanh(1 / w)
    else:
        raw = np.tanh(np.exp(-4 * (x ** 2) / L ** 2) / w) / np.tanh(1 / w)
    return raw


def barrier7(x, w, L):
    w = w
    return np.tanh(1 / w * (2 / w + np.e) ** (-4 * (x ** 2) / L ** 2)) / np.tanh(1 / w)


def barrier6(x, w, L):
    raw = np.tanh((2 / w + np.e) ** (-4 * (x ** 2) / L ** 2) / w) / np.tanh(1 / w)
    return raw


def arg(w):
    if w <= 2 / np.exp(1):
        return np.log(2 / w)
    else:
        return 1


def constants(x, w):
    bar = barrier2(x, w)
    print(f"w={w}")
    print(f"-inf = {bar[0]}")
    print(f"+inf = {bar[-1]}")
    print(f"0 = {max(bar)}")
    print(f"Norm = {simps(barrier2(x, w), x)}")


x = np.linspace(-10, 10, 10000)
w_list = [10 ** -100, 0.01, 0.1, 1, 100]
w_list = [10 ** -100, 0.1, 100]

# w_list = [0.1, 1,1000]
# w_list = np.logspace(-6, 1, num=100)

L = 6
####Plot many barriers
# for i in w_list:
#     plt.plot(x, barrier6(x, i, L), label=f"{i}")
# plt.legend()
# plt.title("Interpolating Barrier for various w")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.savefig("Correct_Barriers.png")
# plt.show()

###### Fitting
# w = 1000
# xm = np.linspace(-10, 10, 50)
# ys = barrier6(xm, w, L)
# popt, pcov = curve_fit(gaussfit, xm, ys)
# fitted = gaussfit(x, popt[0], popt[1], popt[2])
# plt.plot(x, fitted, linestyle="-", label="Fitted", color="k")
# plt.scatter(xm, ys, label=f"Barrier", marker="x")
# plt.legend()
# plt.title(f"Fitting a gaussian to w={w} barrier.png")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.savefig("Correct_Fitting")
# plt.show()

#### Limit Mess
# print(arg(2/np.exp(1)))
# y = [arg(i) for i in w_list]
# plt.plot(w_list,y)
# plt.show()

##### Better Variables
w_list = [-1, 0, 1, 10, 100]
w_list = [10 ** -100, 0.01, 100]
for w in w_list:
    bar = barrier7(x, w, L)
    plt.plot(x, bar)
plt.show()
