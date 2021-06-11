import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def tanhBasic(x, a, b, V):
    return (V / 2) * (np.tanh((x + a) / b) - np.tanh((x - a) / b)) / (np.tanh(a / b))


def tanhAlt(x, a, b, V):
    return V * 2 * np.cosh(a / b) ** 2 / (np.cosh(2 * a / b) + np.cosh(2 * x / b))


def cos(x, L):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < -L or v > L:
            temp[n] = 0
        else:
            temp[n] = 0.5 * (1+np.cos(np.pi * v / L))
    return temp


x = np.linspace(-20, 20, 1000)
V = 10
a = 5
b = 1 * a

# plt.plot(x, tanhBasic(x, a, b, V))
# plt.plot(x, tanhAlt(x, a, b, V), linestyle="--")
# plt.show()

plt.plot(x,cos(x,a))
plt.show()