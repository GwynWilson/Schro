import PhD.Stochastic.Ito_Process as ito
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, rfft, rfftfreq
from scipy import signal


def gauss(t, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (t - mu) ** 2 / (2 * sigma ** 2))


t = 10
dt = 10 ** (-4)
n = int(t / dt)
print(n)
time = [i * dt for i in range(n)]

mu = 0
sig = 1
x = ito.randGauss(mu, sig, n)
print(np.sqrt(np.var(x)))
print(np.mean(x))

t = 10

sig1 = 0
sig2 = 0
sig3 = 0

##### Checking sigma bounds
# for i in x-mu:
#     if abs(i) < 3 * sig:
#         if abs(i) < 2 * sig:
#             if abs(i) < sig:
#                 sig1 += 1
#             sig2 += 1
#         sig3 += 1
#
# print(sig1 / n, sig2 / n, sig3 / n)
#
# plt.scatter(time, x)
# plt.hlines(mu + sig, time[0], time[-1])
# plt.hlines(mu + 2 * sig, time[0], time[-1])
# plt.hlines(mu + 3 * sig, time[0], time[-1])
# plt.show()


##### Hist
# xm = max(x)
# xl = min(x)
# xt = np.linspace(xl,xm,1000)
# plt.hist(x,bins=30,density=True)
# plt.plot(xt,gauss(xt,mu,sig))
# plt.plot()
# plt.title(fr"Gaussian Distribution, $\mu$={mu},$\sigma$={sig}")
# plt.savefig("Gaussian Distribution")
# plt.show()

##### Fourier
# print(rfftfreq(n,dt))
# plt.plot(rfftfreq(n,dt)[1:],rfft(x)[1:])
# plt.show()
# plt.hist(rfft(x)[1:])
# plt.show()

###### Power Density?
# f, pwr = signal.periodogram(x, 1 / dt)
# plt.semilogy(f, pwr)
# plt.xlim([f[1], f[-1]])
# plt.title("Power spectra for gaussian sampling")
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# # plt.savefig("Power Spectra Gauss")
# plt.show()


######Gauss squared
x = sig * np.random.randn(n) + mu
xs = x**2
# plt.plot(time,xs)
# plt.show()

f, pwr = signal.periodogram(x, 1 / dt)
f2, pwr2 = signal.periodogram(xs, 1 / dt)
plt.semilogy(f2,pwr2,label="x^2")
plt.semilogy(f, pwr,label="x")

plt.xlim([f[1], f[-1]])
plt.title("Power spectra for gaussian sampling")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.legend()
# plt.savefig("Power Spectra Gauss")
plt.show()
