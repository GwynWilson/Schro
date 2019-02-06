import numpy as np
from scipy.fftpack import fft, ifft
from scipy.integrate import simps


class Schrodinger(object):
    def __init__(self, x, psi, v, k, hbar=1, m=1, t=0):
        # Setting necessary variables
        self.x = x
        self.dx = x[1] - x[0]
        self.N = len(x)
        self.a = self.N * self.dx

        self.psi_x = psi
        self.psi_k = fft(psi)
        self.v = v
        self.psi_squared = self.mod_square_x(r=True)

        self.hbar = hbar
        self.m = m

        self.t = t
        self.dt = None

        self.dk = k[1] - k[0]
        self.k0 = k[0]
        self.k = k

    def evolve_x(self):
        psi_t = np.zeros(len(self.psi_x), dtype=complex)
        for n, val in enumerate(self.psi_x):
            psi_t[n] = val * (np.exp(-1j * (self.v[n] * self.dt) / self.hbar))
        self.psi_x = psi_t

    def evolve_k(self):
        psi_t = np.zeros(len(self.psi_k), dtype=complex)
        for n, val in enumerate(self.psi_k):
            psi_t[n] = val * (np.exp(-0.5j * (self.hbar * (self.k[n] ** 2) * self.dt) / self.m))
        self.psi_k = psi_t

    def evolve_t(self, N_steps=1, dt=0.1):
        self.dt = dt
        for i in range(N_steps):
            self.evolve_x()
            self.psi_k = fft(self.psi_x)
            #            self.psi_k = fftshift(fft(self.psi_x))
            self.evolve_k()
            self.psi_x = ifft(self.psi_k)
        #            self.psi_x = ifft(fftshift(self.psi_k))
        self.t += (N_steps * self.dt)

    def mod_square_x(self, r=False):
        self.psi_squared = np.real(self.psi_x * np.conj(self.psi_x))
        if r == True:
            return self.psi_squared

    def norm_x(self):
        self.mod_square_x()
        return simps(self.psi_squared, self.x)

    def expectation_x(self):
        self.mod_square_x()
        y = self.psi_squared * self.x
        return simps(y, self.x)

    def expectation_x_square(self):
        self.mod_square_x()
        y = self.psi_squared * self.x * self.x
        return simps(y, self.x)
