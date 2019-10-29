import numpy as np
# from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps, trapz
import matplotlib.pyplot as plt


class Schrodinger(object):
    """
    Class that carries out the evolution of the wave function. Contains important bits of data (such as wave function
    as well as k and x points) and various functions to both carry out the evolution and return calculated variables.
    """

    def __init__(self, x, psi, v, k=None, hbar=1, m=1, t=0, args=None, pi=None):
        """
        :param x: Array of the x points
        :param psi: Initial Wavefunction in x space
        :param v: Potential, can either be given as a function or as an array.
        :param k: Points in K space (may not be necessary but kept for older versions of the code)
        :param hbar: h bar, set to 1 by default
        :param m: mass, set to 1 by default
        :param t: initial time, set to 1 by default
        :param args: Arguments that the potential may require. Inputted in a tuple
        """

        # Setting necessary variables
        self.x = x
        self.dx = x[1] - x[0]
        self.N = len(x)
        self.x_length = self.N * self.dx
        self.deltax = None

        self.psi_x = psi
        self.psi_k = fftshift(fft(psi))
        self.normalise_x()
        self.psi_squared = self.mod_square_x(r=True)
        self.potential = None

        self.t = t
        self.dt = None

        # Code for accounting for the potential being a function (necessary for time dependence)
        self.call = False
        self.args = args
        if callable(v):
            self.potential = v
            if self.args != None:
                self.v = self.potential(self.x, self.t, args=self.args)
            else:
                self.v = self.potential(self.x, self.t)
            self.call = True
        else:
            self.v = v

        self.hbar = hbar
        self.m = m

        self.pi = pi
        if self.pi == True:
            self.k = fftfreq(self.N, self.dx)
        else:
            self.k = fftfreq(self.N, self.dx / (2 * np.pi))

        k_lim = np.pi / self.dx
        self.k1 = -k_lim + (2 * k_lim / self.N) * np.arange(self.N)
        self.k = self.k1
        self.dk = self.k[1] - self.k[0]

        self.mod_square_k()

    def evolve_x(self, half=False):
        """
        Function to evolve the wave function in x space
        :return: None
        """

        # Code to account for the potential being a function
        if self.call:
            if self.args != None:
                self.v = self.potential(self.x, self.t, args=self.args)
            else:
                self.v = self.potential(self.x, self.t)

        # Evolving the wave function
        if half:
            self.psi_x = self.psi_x * np.exp(-0.5j * (np.asarray(self.v) * self.dt) / self.hbar)
        else:
            self.psi_x = self.psi_x * np.exp(-1j * (np.asarray(self.v) * self.dt) / self.hbar)

    def evolve_k(self):
        """
        Function to evolve the wavefunction in K space
        :return: None
        """

        if self.pi == True:
            self.psi_k = self.psi_k * (
                np.exp(-0.5j * (self.hbar * (2 * np.pi * np.asarray(self.k) ** 2) * self.dt) / self.m))
        else:
            # self.psi_k = self.psi_k * (np.exp(-0.5j * (self.hbar * (np.asarray(self.k) ** 2) * self.dt) / self.m))
            self.psi_k = self.psi_k * (
                np.exp(-1j * (self.hbar * (np.asarray(self.k) ** 2) * self.dt) / (2 * self.m)))

    def evolve_t(self, N_steps=1, dt=0.1):
        """
        Function to carry out the time evolution of the wave function
        :param N_steps: Number of steps
        :param dt: Small time step
        :return:
        """
        self.dt = dt
        for i in range(N_steps):
            self.evolve_x(half=True)
            self.psi_k = fftshift(fft(self.psi_x, norm="ortho"))
            # self.psi_k = self.normalise_k()
            self.evolve_k()
            self.psi_x = ifft(fftshift(self.psi_k), norm="ortho")
            self.evolve_x(half=True)
            self.psi_x = self.normalise_x()

        self.t += (N_steps * self.dt)

    def mod_square_x(self, r=False):
        """
        Function that will give the mod square of the wave function in x space
        :param r: Will return the mod square of the wave function as an array, set to not do this by default
        :return: Returns if r= True
        """

        self.psi_squared = np.real(self.psi_x * np.conj(self.psi_x))

        if r == True:
            return self.psi_squared

    def mod_square_k(self, r=False):
        """
        Function that will give the mod square of the wave function in k space
        :param r: Will return the mod square of the wave function as an array, set to not do this by default
        :return: Returns if r= True
        """
        self.psi_squared_k = np.real(self.psi_k * np.conj(self.psi_k))

        if r == True:
            return self.psi_squared_k

    def normalise_k(self):
        """
        Function normalising the wave function in k space
        :return: Returns the normalised wave function in k space
        """
        k_s = self.mod_square_k(r=True)
        norm = simps(k_s, self.k)
        self.psi_k = self.psi_k / np.sqrt(norm)
        self.mod_square_k()
        return self.psi_k

    def normalise_x(self):
        """
        Function normalising the wave function in k space
        :return: Returns the normalised wave function in k space
        """
        x_s = self.mod_square_x(r=True)  # Getting the mod square of the wave function
        norm = simps(x_s, self.x)
        self.psi_x = self.psi_x / np.sqrt(norm)
        self.mod_square_x()
        return self.psi_x

    def norm_x(self):
        """
        Outputting what the normalisation of the wave function is. Used for tracking this over time.
        :return: Normalisation of the wavefunction
        """
        self.mod_square_x()
        return simps(self.psi_squared, self.x)

    def expectation_x(self):
        """
        :return: Returns expectation value of x
        """
        self.mod_square_x()
        y = self.psi_squared * self.x
        return simps(y, self.x)

    def expectation_k(self):
        self.mod_square_k
        return simps(self.psi_squared_k * self.k, self.k)

    def expectation_x_square(self):
        """
        :return: Returns expectation value of x^2
        """
        self.mod_square_x()
        y = self.psi_squared * self.x * self.x
        return simps(y, self.x)

    def barrier_transmition(self):
        """
        Function that returns te tunelling probability past the end of a potential barrier. For this to work the end
        of the barrier needs to be passed as an argument when defining the class. The barrier also needs to be placed
        on the x space lattice.
        :return: Mod square of wave function beyond the barrier
        """
        self.mod_square_x()
        end = self.args
        ind = np.where(self.x == end)[0][0]  # Index of x point where the barrier ends
        return simps(self.psi_squared[ind:], self.x[ind:])  # Integrates from beyond the end point of the barrier

    def energy(self):
        """
        Function to calculate the energy of a system. Takes potential energy in x space and kinetic from k space
        DOES NOT CURRENTLY WORK
        :return: Total energy of a system
        """

        self.normalise_x()
        x_sp = self.psi_squared * np.asarray(self.v)
        x_e = simps(x_sp, self.x)  # Energy from potential in x space

        self.normalise_k()
        # plt.plot(self.k,(np.asarray(self.k) ** 2))
        # plt.show()
        k_sp = self.psi_squared_k * ((self.hbar ** 2) / (2 * self.m)) * (np.asarray(self.k) ** 2)
        # plt.plot(self.k, k_sp)
        # plt.show()
        k_e = simps(k_sp, self.k)  # Energy from potential in k space

        return x_e + k_e

    def x_width(self):
        exs = self.expectation_x_square()
        ex = self.expectation_x()
        self.deltax = np.sqrt(exs - ex ** 2)
        return self.deltax

    def theoreticalEnergy(self, k=None):
        if self.deltax == None:
            self.x_width()
        return self.hbar ** 2 / 2 * self.m * (k ** 2 + 1 / (4 * self.deltax ** 2))

    def impedence(self, E=None):
        if E == None:
            E = self.energy()
        for n, i in enumerate(reversed(self.v)):
            diff = (E - i)
            if diff == 0:
                diff += 10 ** -99
            K = 1j * np.sqrt(2 * self.m * diff + 0j) / self.hbar
            z0 = -1j * self.hbar * K / self.m
            if n == 0:
                zload = z0
            else:
                zload = zin
            zin = z0 * ((zload * np.cosh(K * self.dx) - z0 * np.sinh(K * self.dx)) / (
                    z0 * np.cosh(K * self.dx) - zload * np.sinh(K * self.dx)))

        coeff = np.real(((zin - z0) / (zin + z0)) * np.conj((zin - z0) / (zin + z0)))
        return 1 - coeff

    def impedencePacket(self, tol=10 ** -9):
        self.normalise_k()
        self.mod_square_k()
        T_tot = 0
        for k, w in zip(self.k, self.psi_squared_k):
            if w > tol:
                E = (self.hbar ** 2 * k ** 2) / (2 * self.m)
                imp = self.impedence(E=E)
                T_tot += imp * w * self.dk
        else:
            return T_tot
