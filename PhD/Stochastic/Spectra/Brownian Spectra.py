import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit


def brownian(n, dt):
    w = np.random.normal(size=n, loc=0, scale=np.sqrt(dt))
    return np.cumsum(w)


def expected(f, s):
    return s / f ** 2


n = 1000000
dt = 0.00001
tl = [i * dt for i in range(n)]
ws = brownian(n, dt)
# plt.plot(tl, ws)
# plt.show()


# plt.plot(rfftfreq(n,dt)[1:],rfft(ws)[1:])
# plt.show()
# plt.hist(rfft(ws)[1:])
# plt.show()

f, pwr = signal.periodogram(ws, 1 / dt)

popt, pcov = curve_fit(expected, f[1:], pwr[1:])
print(popt)

plt.loglog(f, pwr)
expected = [popt[0] / (i ** 2) for i in f]
plt.loglog(f, expected)
plt.xlim([f[1], f[-1]])
plt.title("Power spectra for gaussian sampling")
plt.xlabel("Frequency")
plt.ylabel("Power")
# plt.savefig("Power Spectra Gauss")
plt.show()
