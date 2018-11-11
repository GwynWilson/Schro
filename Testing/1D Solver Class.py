import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from matplotlib import animation


class schrodinger(object):
    def __init__(self, x, psi, v, k, hbar=1, m=1, t=0):
        # Setting neccesary variables
        self.x = x
        self.dx = x[1] - x[0]
        self.N = len(x)
        self.a = self.N * self.dx

        self.psi_x = psi
        self.psi_k = fft(psi)
        self.v = v
        self.hbar = hbar
        self.m = 1

        self.t = t
        self.dt = None

        self.dk = k[1] - k[0]
        self.k0 = k[0]
        self.k = k

    def evolve_x(self):
        psi_t = np.zeros(len(self.psi_x), dtype=complex)
        for n, val in enumerate(self.psi_x):
            psi_t[n] = val * (-np.exp(1j * (self.v[n] * self.dt) / self.hbar))
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


def gauss_init(x, k0, x0=0, d=1):
    # initalised gausian wavefunction
    return 1/np.sqrt(2*np.pi*d) * np.exp(-((x - x0) ** 2) / (2*d)) * np.exp(1j * k0 * x)


def V(x):
    # zero potential
#    return 0

    if x > -30 and x < -29.5:
        return 10**30
    else:
        return 0



def conj(v):
    return np.real(v * np.conj(v))


# Defining x axies
N = 2 ** 10
dx = 0.1
a = dx * N
x = dx * (np.arange(N) - 0.5 * N)
xmax = -x[0]
x0 = -0.75 * xmax


# Defining k axies
dk = 2 * np.pi / a
k0 = - 0.5 * N * dk
k = k0 + dk * np.arange(N)
ks = fftshift(k)

# Initial k value
k_ini = 20

hbar = 1
m = 1

# Defining time quantities
t = 0
dt = 0.01
Nstep = 1
t_max = 120
frames = int(t_max / float(Nstep * dt))

# Defining wavefunction and potential
psi_x = gauss_init(x, k_ini, x0, d=1)
v_x = [V(k) for k in x]

print(np.sum([abs(k)*dx for k in psi_x]))

s = schrodinger(x, psi_x, v_x, ks, hbar, m, t)

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
sin_line, = ax1.plot([], [])
potential_line, = ax1.plot([], [])
centre_line, = ax1.plot([], [])
actual_line, = ax1.plot([], [])

ax1.set_xlim(-xmax, xmax)
ax1.set_ylim(-0.2, 0.2)

ax2 = fig.add_subplot(212)
k_line, = ax2.plot([], [])
ax2.set_xlim(k[0], k[N - 1])
ax2.set_ylim(-50, 50)


def init():
    sin_line.set_data(x, conj(s.psi_x))
    k_line.set_data(k, conj(s.psi_k))
    centre_line.set_data([], [])
    actual_line.set_data([], [])
    potential_line.set_data(x, v_x)
    return sin_line, k_line, centre_line, actual_line,


def animate(i):
    s.evolve_t(Nstep, dt)
    sin_line.set_data(s.x, conj(s.psi_x))
    k_line.set_data(s.k, abs(s.psi_k))
    centre_line.set_data(2 * [x0 + (s.t * (k_ini * hbar / m))], [0, 1])
    potential_line.set_data(x, v_x)
    print(s.t)
    return sin_line, k_line, centre_line, actual_line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)
plt.show()
