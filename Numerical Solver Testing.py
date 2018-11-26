import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation

from Schrodinger_Solver import Schrodinger
from Animate import Animate


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(1j * k0 * x)


def x_pos(t, x0, kini, hbar=1, m=1):
    return x0 + t * kini * (hbar / m)


# Defining x axis
N = 2 ** 11
dx = 0.1
x_length = N * dx
x = np.linspace(0, x_length, N)
x0 = 50

# Defining Psi and V
k_initial = 10
psi_x = gauss_init(x, k_initial, x0, d=1)
V_x = np.zeros(N)

# Defining K range
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Defining time steps
t = 0
dt = 0.01
step = 10

sch = Schrodinger(x, psi_x, V_x, k)

plt.plot(sch.x, sch.mod_square_x(True))
plt.show()


a = Animate(sch, V_x, step, dt, lim1=((0, x_length), (0, max(psi_x))), lim2=((ks[0], ks[N - 1]), (0, 50)))
a.make_fig()

"""
frames = int(120 / float(step * dt))
# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
sin_line, = ax1.plot([], [])
potential_line, = ax1.plot([], [])
centre_line, = ax1.plot([], [])
actual_line, = ax1.plot([], [])

ax1.set_xlim(0, x_length)
ax1.set_ylim(0, 0.2)

ax2 = fig.add_subplot(212)
k_line, = ax2.plot([], [])
ax2.set_xlim(ks[0], ks[N - 1])
ax2.set_ylim(-50, 50)


def init():
    sin_line.set_data(x, sch.mod_square_x(True))
    k_line.set_data([], [])
    centre_line.set_data([], [])
    actual_line.set_data([], [])
    potential_line.set_data(x, V_x)
    return sin_line, k_line, centre_line, actual_line,


def animate(i):
    sch.evolve_t(step, dt)
    sin_line.set_data(sch.x, sch.mod_square_x(True))
    k_line.set_data(sch.k, abs(sch.psi_k))
    centre_line.set_data(2 * [x0 + (sch.t * k_initial)], [0, 1])
    potential_line.set_data(x, V_x)
#    print(sch.t)
    return sin_line, k_line, centre_line, actual_line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)
plt.show()

"""

"""

t_list = []
norm_x = []
expec_x = []
expec_xs = []
print(sch.norm_x())

plt.plot(sch.x, sch.mod_square_x(True))
plt.xlabel('x')
plt.ylabel('Psi(x)^2')
plt.show()

for i in range(100):
    if i != 0:
        sch.evolve_t(step, dt)
    t_list.append(sch.t)
    norm_x.append(sch.norm_x() - 1)
    expec_x.append(sch.expectation_x())
    expec_xs.append(sch.expectation_x_square() - expec_x[i] ** 2)

plt.plot(t_list, norm_x, linestyle='none', marker='x')
plt.title('Normalistaion of wavefunction over time')
plt.xlabel('Time')
plt.ylabel('Normalisation-1')
plt.savefig('Normalisation.png')
plt.show()


plt.plot(t_list, expec_x, label='Calculated x')
plt.plot(t_list, [x_pos(j, x0, k_initial) for j in t_list], linestyle='--', label='Expected x')
plt.title('Expectation value of x over time')
plt.xlabel('Time')
plt.ylabel('<x>')
plt.legend(loc='best', fancybox=True)
plt.savefig('Expec_X.png')
plt.show()


plt.plot(t_list, expec_xs)
plt.title('Width of distribution over time')
plt.xlabel('Time')
plt.ylabel('<delta x>')
plt.savefig('delta_x.png')
plt.show()
"""
