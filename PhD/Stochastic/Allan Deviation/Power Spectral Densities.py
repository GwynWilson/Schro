import numpy as np
import allantools as at
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import periodogram

from PhD.Stochastic.Heller import Heller


def poly1(x, m, c):
    x=np.asarray(x)
    return m * x + c


def brown(x, b2):
    return b2 / x ** 2


def whiteNoise(n, r, b0):
    WN = at.noise.white(n, b0, r)
    freq, psd = at.noise.numpy_psd(WN, r)

    coeff, cov = curve_fit(poly1, freq, psd, (1, b0))
    fitline = poly1(freq, coeff[0], coeff[1])
    print("Estimated b0", coeff[1])

    plt.plot(freq, psd)
    plt.plot(freq, fitline)
    plt.plot()
    plt.show()
    return 0


def brwonNoise(n, r, b2):
    BN = at.noise.brown(n, b2, r)
    freq, psd = at.noise.scipy_psd(BN, r, nr_segments=4)

    coeff, cov = curve_fit(brown, freq[1:], psd[1:])
    fitline = brown(freq, coeff[0])
    print("Estimated b2", coeff[0])

    plt.loglog(freq, psd)
    plt.loglog(freq, fitline)
    plt.plot()
    plt.show()
    return 0


def brownAv(n, r, b2, n_run):
    t = [i * 1 / r for i in range(n)]
    av = np.zeros(n)
    for i in range(n_run):
        bn = at.noise.brown(n, b2, r)
        av += bn ** 2 / n_run
    coef, cov = curve_fit(poly1, t, av)
    fitline = poly1(t, coef[0], coef[1])
    print("sig^2", coef[0])
    plt.plot(t, av)
    plt.plot(t,fitline)
    plt.show()
    return 0


n = 10000
dt = 1
r = 1 / dt

n_run = 10000

b0 = 100
b2 = 10

whiteNoise(n, r, b0)
# brwonNoise(n,r,b2)
# brownAv(n, r, b2, n_run)
