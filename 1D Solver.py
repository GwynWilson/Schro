import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


def gauss_init(x, k0):
    # initalised gausian wavefunction
    return np.exp(-((x - x0) ** 2)) * np.exp(1j * k0 * x)


def V(x):
    # zero potential
    return 0


def evolvex(x, psi):
    # step to evolve the x wavefunction
    psi_t = np.zeros(len(psi), dtype=complex)
    for n, v in enumerate(psi):
        psi_t[n] = v * (np.exp(-1j * (V(x[n]) * dt) / hbar))
    return psi_t


def evolvek(k, psi):
    # step to evolve the k wavefunction
    psi_t = np.zeros(len(psi), dtype=complex)
    for n, v in enumerate(psi):
        psi_t[n] = v * (np.exp(-0.5j * (hbar * (k[n] ** 2) * dt) / m))
    return psi_t


def plot_xs(psi, V):
    # plotting the mod square of the wavefunction and the potential
    plt.xlim(min(xrange), max(xrange))
    plt.plot(xrange, psi, label='psi^2')
    plt.plot(xrange, v_x, label='V(x)')
    plt.plot(2 * [x0 + t * (-p0 / m)], [0, 1]) # plotting a line where the peak should be
    plt.show()


def mod_square(psi):
    # returns mod square of wavefunction
    return np.real(psi * np.conj(psi))


def cycle(N_s, psi, t):
    for i in range(N_s):
        psi = evolvex(xrange, psi)
        psik = fft(psi_x)
        psik = evolvek(krange, psik)
        psi = ifft(psik)
    t += (N_s * dt)
    return psi, t


# Defining x variables
N = 2**11
dx = 0.1
a = dx * N
xrange = dx * (np.arange(N) - 0.5 * N)
xmax = -xrange[0]


# Defining p variables
#hbar = 10 ** -34
hbar = 1
m = 1
#p0 = np.sqrt(2 * m) #initial momentum?
x0 = -0.75 * xmax

#k0 = p0 / hbar
dk = 2 * np.pi / a #not sure why
k0 = - 0.5 * a
p0 = k0/hbar
krange = k0 + dk * np.arange(N) #generating list of k


# creating wavefunction and potential
psi_x = np.array([gauss_init(n, -k0) for n in xrange])
v_x = [V(i) for i in xrange]


dt = 0.1
t = 0
N_s = 1

psi = np.real(psi_x * np.conj(psi_x))
plot_xs(psi, v_x)

# Testing a loop
for i in range(50):
    for n in range(N_s):
        psi_x = evolvex(xrange, psi_x)
        psi_k = fft(psi_x)
        psi_k = evolvek(krange, psi_k)
        psi_x = ifft(psi_k)
    t += dt * N_s
    psi = np.real(psi_x * np.conj(psi_x))
    plot_xs(psi, v_x)
