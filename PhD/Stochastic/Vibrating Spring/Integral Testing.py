from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


def integrand(t):
    return np.exp(g * t) * np.cos(w * (2 * T - 2 * t))


def firstResult(t):
    pref = 1/(g**2+4*w**2)
    return pref*(g*np.exp(g*t) -g*np.cos(2*w*t) + 2*w*np.sin(2*w*t))


T = 1
g = 1
w = 10

T_list = np.linspace(0, 5, 1000)
result = []
for T in T_list:
    result.append(quad(integrand, 0, T)[0])
plt.plot(T_list, result)
plt.plot(T_list, firstResult(T_list))
plt.show()
