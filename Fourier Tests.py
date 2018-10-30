import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft, ifft, fftfreq, fftshift

def sin(x, f, t):
    return np.cos(f*x + t)

class sinevolve(object):
    def __init__(self, x, y, f, dt, t, k):
        self.x = x
        self.dx = x[1] -x[0]
        self.N = len(x)
        self.sinx = y
        self.sink = fft(self.sinx)
        self.f = f
        self.t = t
        self.dt = dt
        self.k = k
        self.dk = k[1] - k[0]

    def evolvt(self, dt, Nstep):
        for i in range(Nstep):
            self.sinx = sin(self.x, self.f, self.t)
            self.t += self.dt
            self.sink = fft(self.sinx)

    def get_k(self):
        return self.sink

    def get_k_range(self):
        return self.k

    def get_t(self):
        return self.t

    def get_y(self):
        return self.sinx

    def get_x(self):
        return self.x


#Setting up x
N = 2**15
dx = 0.1
a = dx * N
#x = dx * (np.arange(N) - 0.5 * N)
x = np.linspace(0, N*dx, N)
xmax = -x[0]


#dk = 2*np.pi / a
#k = -(a/2) + dk * np.arange(N)
k = fftfreq(N, dx/(2*np.pi))
k = fftshift(k)


#Setting up time
f = 0.5
t = 0
dt = 0.1
Nstep = 10
t_max = 120
frames = int(t_max / float(Nstep * dt))

sinx = [sin(j, f, t) for j in x]
s = sinevolve(x, sinx, f, dt, t, k)

#Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
sin_line, = ax1.plot([], [])

ax1.set_xlim(x[0], 100)
ax1.set_ylim(-1, 1)

ax2 = fig.add_subplot(212)
k_line, = ax2.plot([], [])
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)

def init():
    sin_line.set_data(x, sinx)
    k_line.set_data(s.get_k_range(), s.get_k())
    return sin_line, k_line,

def animate(i):
    s.evolvt(dt, Nstep)
    sin_line.set_data(s.get_x(), s.get_y())
    k_line.set_data(s.get_k_range(), 1/N * abs(fftshift(s.get_k())))
    return sin_line, k_line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)
plt.show()
