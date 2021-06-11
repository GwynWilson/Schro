import allantools as at
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def straight(x, m):
    x = np.asarray(x)
    return m * x


def straightC(x, c):
    return -0.5 * x + c


def shitderiv(x, y):
    i = len(y)
    deriv = []
    for n in range(i - 1):
        deriv.append((y[n + 1] - y[n]) / (x[n + 1] - x[n]))
    return np.asarray(deriv)


def findIntercept(logt, logdev, slope):
    deiv = shitderiv(logt, logdev)
    wind = (np.abs(deiv - slope)).argmin()
    return logdev[wind] - slope * logt[wind]


n = 10000
dt = 1
t = [i * dt for i in range(n)]
r = 1 / dt  # Sample rate in Hz

tau = np.logspace(0, 3, 1000)  # Tau

N = 100

########## Is the noise correct?
# ws = np.zeros(n)
# for i in range(N):
#     y = at.noise.white(n, b0=20, fs=r)
#     ws += np.cumsum(y * dt) ** 2 / N
#
# popt, covt = curve_fit(straight, t, ws)
# print(popt)
#
# plt.plot(t, ws)
# plt.plot(t, straight(t, popt[0]))
# plt.show()


########### Alan dev
b2 = 0.2
b0 = 100
sig = 100
y = at.noise.brown(n, b2=b2, fs=r)
x = at.noise.white(n, b0=b0, fs=r)
# x = sig * np.random.randn(n) / np.sqrt(dt)
z = 10 * at.noise.pink(n)

# axis, psd = at.noise.numpy_psd(y, r)
# plt.loglog(axis, psd)
#
# axis, psd = at.noise.numpy_psd(x, r)
# plt.loglog(axis, psd)
#
# axis, psd = at.noise.numpy_psd(z, r)
# plt.loglog(axis, psd)
# plt.show()



taus, adevs, err, ns = at.oadev(x, r, data_type="freq", taus=tau)
logt = np.log(taus)
logdev = np.log(adevs)

# Fitting for white noise
slope = -0.5
w = findIntercept(logt, logdev, slope)
wline = logt * slope + w

wind = (np.abs(logt - 1)).argmin()
logN = wline[wind]

coef, cov = curve_fit(straightC, logt, logdev)
fitline = straightC(logt, coef[0])
logNfit = fitline[wind]

print(sig, np.exp(logN)**2, np.exp(logNfit)**2)  # intercept gives psd

# Fitting for brown noise

# taus, adevs, err, ns = at.oadev(y, r, data_type="freq", taus=tau)
# logt = np.log(taus)
# logdev = np.log(adevs)
# slope = 0.5
# b = findIntercept(logt, logdev, slope)
# bline = logt * slope + b
# bind = (np.abs(logt - 3)).argmin()  # parameter found at t=3
# print(0.2, (np.exp(bline[bind]) / (2 * np.pi))**2)
#
# coeff,cov = curve_fit(straightC,logt,logdev)
# fitline = straightC(logt,coeff[0])

plt.plot(logt, logdev)
plt.plot(logt, wline, linestyle=":")
# plt.plot(logt, bline, linestyle=":")
plt.plot(logt, fitline)
plt.ylim((min(logdev), max(logdev)))
plt.show()
