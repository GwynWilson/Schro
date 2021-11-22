from Heller import Heller
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def derivsStoc(t, current, args, eta, dt):
    dtp = args[0]
    sig = args[1]
    x = eta ** 2
    v = 0
    a = 0
    g = 0
    return x, v, a, g


def poly(x, m, c):
    return m * x + c


def noise(n, dt):
    return np.random.randn(n) / np.sqrt(dt)


def gauss(r, sig, mu):
    return np.exp(-((r - mu) ** 2) / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)


n = 100
dt = 0.01
t = n * dt
init = [0, 0, 0, 0]
sig = 10
args = (dt, sig)

####### dt dependence?
# solverStoc1 = Heller(n, dt, init, derivsStoc)
# solverStoc1.averageRuns(100, args)
# popt1, covt1 = curve_fit(poly, solverStoc1.tl, solverStoc1.x_av)
# print(popt1[0])
#
# solverStoc2 = Heller(2 * n, dt / 2, init, derivsStoc)
# solverStoc2.averageRuns(100, args)
# popt2, covt2 = curve_fit(poly, solverStoc2.tl, solverStoc2.x_av)
# print(popt2[0])
#
# solverStoc3 = Heller(4 * n, dt / 4, init, derivsStoc)
# solverStoc3.averageRuns(100, args)
# popt3, covt3 = curve_fit(poly, solverStoc3.tl, solverStoc3.x_av)
# print(popt3[0])
#
# plt.plot(solverStoc1.tl, solverStoc1.x_av, label=f"{dt}")
# plt.plot(solverStoc2.tl, solverStoc2.x_av, label=f"{dt/2}")
# plt.plot(solverStoc3.tl, solverStoc3.x_av, label=f"{dt/4}")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Integrand")
# plt.title(r"$\eta^2$ integral for varying dt (100 runs)")
# plt.savefig("Eta Squared Test 2")
# plt.show()


########## dt nice plot
# n_list = np.linspace(100, 1000, 50,dtype=int)
# T = 1
# m_list = []
# for n_i in n_list:
#     if n_i % 100 == 0:
#         print(n_i)
#     solverStoc = Heller(n_i, T / n_i, init, derivsStoc)
#     solverStoc.averageRuns(50, args)
#     popt, covt = curve_fit(poly, solverStoc.tl, solverStoc.x_av, (n_i, 0))
#     m_list.append(popt[0])
#
# plt.title(r"Gradient of $\eta^2$ integral (50 runs)")
# plt.scatter(n_list, m_list,marker="x")
# plt.ylabel("Gradient of integral")
# plt.xlabel("Number of points (n)")
# plt.savefig("eta squared grad")
# plt.show()


############# dt var
dt = 0.01
N = 1000000
x = noise(N, dt)
xs = x ** 2
r = np.linspace(min(x), max(x), N)
sig = 1 / np.sqrt(dt)

# y = np.exp(-(r ** 2) / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
# plt.hist(xs, bins=30, density=True)
# plt.show()

Nsample = 1000
cuts = np.array_split(xs, N / Nsample)
means = []
for i in cuts:
    means.append(np.mean(i))

r = np.linspace(min(means), max(means), N)

z = plt.hist(means, bins=30, density=True,label="Hist")
popt, covt = curve_fit(gauss, z[1][:-1], z[0],(np.sqrt(2) * sig ** 2,sig ** 2))
plt.plot(r, gauss(r, popt[0], popt[1]), label="Fit")
plt.plot(r, gauss(r, np.sqrt(2) * sig ** 2 / np.sqrt(Nsample), sig ** 2), label="Theory")
plt.legend()
plt.xlabel(r"Mean of $\eta^2$")
plt.ylabel(r"Probability Density")
plt.title(f"Central Limit Theorem, dt={dt}")
plt.savefig("Central Limit")
plt.show()

####### Sigma testing
# solverStoc1 = Heller(n, dt, init, derivsStoc)
# solverStoc1.averageRuns(100, args,euler=True)
#
# popt, covt = curve_fit(poly, solverStoc1.tl, solverStoc1.a_av)
#
# print(popt)
# print(np.sqrt(popt[0]), sig)
#
# plt.plot(solverStoc1.tl, solverStoc1.g_av)
# plt.show()
#
# plt.plot(solverStoc1.tl, solverStoc1.a_av)
# plt.show()

############dt dependence Euler
# solverStoc1 = Heller(n, dt, init, derivsStoc)
# solverStoc1.averageRuns(100, args, euler=True)
#
# solverStoc2 = Heller(10 * n, dt / 10, init, derivsStoc)
# solverStoc2.averageRuns(100, args, euler=True)
#
# solverStoc3 = Heller(100 * n, dt / 100, init, derivsStoc)
# solverStoc3.averageRuns(100, args, euler=True)
#
# plt.plot(solverStoc1.tl, solverStoc1.x_av, label=f"{dt}")
# plt.plot(solverStoc2.tl, solverStoc2.x_av, label=f"{dt/10}")
# plt.plot(solverStoc3.tl, solverStoc3.x_av, label=f"{dt/100}")
# plt.legend()
# plt.xlabel("time")
# plt.ylabel("Integrand")
# plt.title("eta squared integral for varying dt Euler")
# plt.savefig("Eta Squared Test Euler")
# plt.show()
