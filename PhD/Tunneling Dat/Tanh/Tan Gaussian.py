from Schrodinger_Solver import Schrodinger
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


# from Input_Parameters_Realistic import *


def gauss(x, mu=0, sig=1):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def barrier(x, w):
    return np.tanh((gauss(x)) / w)


def barrier(x, w):
    return np.tanh((gauss(x)) / w) * 1 / np.tanh((gauss(0)) / w)


def barrier2(x, w):
    raw = np.tanh(np.exp(-x ** 2) / w)
    return raw/simps(raw,x)


def constants(x, w):
    bar = barrier2(x, w)
    print(f"w={w}")
    print(f"-inf = {bar[0]}")
    print(f"+inf = {bar[-1]}")
    print(f"0 = {max(bar)}")
    print(f"Norm = {simps(barrier2(x, w), x)}")


x = np.linspace(-10, 10, 1000)
w_list = [0.001, 0.01, 0.1, 1]

for i in w_list:
    constants(x, i)
    plt.plot(x, barrier(x, i))
    # plt.plot(x,np.tanh(gauss(x)/i))
plt.show()
