import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.animation as animation


def gauss_init(x, k0):
    return np.exp(-((x-x0) ** 2)) * (np.cos(k0 * x) + 1j * np.sin(k0 * x))


def V(x):
    return 0


def evolvex(x, psi):
    psi_t = np.zeros(len(psi), dtype=complex)
    for n, v in enumerate(psi):
        psi_t[n] = v * (np.exp(1j * (V(x[n]) * dt) / hbar))
    return psi_t


def evolvek(k, psi):
    psi_t = np.zeros(len(psi), dtype=complex)
    for n, v in enumerate(psi):
        psi_t[n] = v * (np.exp(-0.5j * (hbar * (k[n] ** 2) * dt) / m))
    return psi_t


def plot_xs(psi, V):

    plt.xlim(min(xrange),max(xrange))
    plt.plot(xrange, psi, label='psi^2')
    plt.plot(xrange, v_x, label='V(x)')
    plt.plot(2 * [x0 + t * (p0 / m)], [0, 1])
    plt.show()


def mod_square(psi):
    return np.real(psi * np.conj(psi))


def cycle(N_s,psi,t):
    for i in range(N_s):
        psi_x = evolvex(xrange, psi)
        psi_k = fft(psi_x)
        psi_k = evolvek(krange, psi_k)
        psi_x = ifft(psi_k)
    t += (N_s*dt)


# Defining x variables
xmax = 10
dx = 0.1
xrange = np.arange(-xmax, xmax, dx)


# Defining p variables
#hbar = 10 ** -34
hbar = 1
m = 1.5
p0 = np.sqrt(2 * m)
x0 = -0.75 * xmax

k0 = p0 / hbar
dk = 2 * np.pi / (2 * xmax)
krange = (k0 + dk) * np.arange(len(xrange))

psi_x = np.array([gauss_init(n, k0) for n in xrange])
v_x = [V(i) for i in xrange]


dt = 0.01
t = 0
N_s = 50
t_m = 120
frames = int(t_m / float(N_s * dt))

fig, ax = plt.subplots()
psi_l, = ax.plot([],[])


def init():
    psi_l.set_data(xrange, mod_square(psi_x))
    return psi_l,


def animate(i):
    cycle(N_s,psi_x,t)
    psi_l.set_ydata(mod_square(psi_x))
    return psi_l,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)
plt.show()

# Testing a loop
# for i in range(5):
#     psi_x, t = cycle(N_s, psi_x, t)
#     psi = np.real(psi_x * np.conj(psi_x))
#     plot_xs(psi, v_x)




