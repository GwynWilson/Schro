import numpy as np
import matplotlib.pyplot as plt


def reflectionProb(v, E, m=1, hbar=1):
    if E < max(v):
        return 1
    for n, i in enumerate(reversed(v)):
        K = 1j * np.sqrt(2 * m * (E - i)) / hbar
        z0 = -1j * hbar * K / m
        if n == 0:
            zload = z0
        else:
            zload = zin
        zin = z0 * ((zload * np.cosh(K * dx) - z0 * np.sinh(K * dx)) / (z0 * np.cosh(K * dx) - zload * np.sinh(K * dx)))

    return abs((zin - z0) / (zin + z0)) ** 2


def refTheory(v0, E):
    if E < v0:
        return 1
    return ((E - np.sqrt(E * (E - v0))) / (E + np.sqrt(E * (E - v0)))) ** 2


def stepv(x, A, x1):
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        else:
            temp[n] = A
    return temp


N = 2 ** 11

m = 1.44 * 10 ** -24
hbar = 1.054 * 10 ** -34
v0 = 1000 * 2 * np.pi * hbar
E = 5000 * 2 * np.pi * hbar
dx = 4 * 10 ** - 5

lim = 3 * 10 ** -3

x = np.arange(-lim, lim, dx)
leng = len(x)
print(leng)
v = stepv(x, v0, x[int(leng / 2)])

El = 5000 * 2 * np.pi * hbar * (np.arange(0, 1, 2 * 10 ** -4))

# plt.plot(x*10**3, v/(2 * np.pi * hbar))
# plt.show()

# for n, i in enumerate(reversed(v)):
#     K = 1j * np.sqrt(2 * m * (E - i)) / hbar
#     z0 = -1j * hbar * K / m
#     if n == 0:
#         zload = z0
#     else:
#         zload = zin
#     zin = z0 * ((zload * np.cosh(K * dx) - z0 * np.sinh(K * dx)) / (z0 * np.cosh(K * dx) - zload * np.sinh(K * dx)))
#
rl = np.array([reflectionProb(v, i, m, hbar) for i in El])
rt = np.array([refTheory(v0, i) for i in El])
plt.plot(El / (2 * np.pi * hbar * 10 ** 3), rl, label="Simulated Reflection", linestyle="", marker="o")
plt.plot(El / (2 * np.pi * hbar * 10 ** 3), rt, label="Theoretical Reflection", linestyle="--")
plt.legend()
plt.ylabel("Reflection Probability")
plt.xlabel("Energy (kHz)")
plt.title("Reflection Probability From Impedence Method")
plt.savefig("Impedence_Method_Ref.png")
plt.show()

plt.plot(El / (2 * np.pi * hbar * 10 ** 3), 1-rl, label="Simulated Reflection", linestyle="", marker="o")
plt.plot(El / (2 * np.pi * hbar * 10 ** 3), 1-rt, label="Theoretical Reflection", linestyle="--")
plt.legend()
plt.ylabel("Transmission Probability")
plt.xlabel("Energy (kHz)")
plt.title("Transmission Probability From Impedence Method")
plt.savefig("Impedence_Method_Trans.png")
plt.show()

# reflection = reflectionProb(v,E,m,hbar)
# print("Reflection Coefficient", reflection)
# print("Transmission Coefficent", 1 - reflection)
