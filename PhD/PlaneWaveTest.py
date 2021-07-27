import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def planeWave(x, t, k, w):
    return np.exp(1j * k * x - w * t)


def planeAngle(x, t, k, w):
    wave = planeWave(x, t, k, w)
    return np.angle(wave)


x = np.linspace(0, 100, 1000)

k = 1
w = 1

t = 0
Nt = 10000
T = 10
dt = T / Nt

####### PlaneWave
# plt.plot(x, np.real(planeWave(x, 0, k, w)))
# plt.plot(x, np.imag(planeWave(x, 0, k, w)))
# plt.show()

# ######## Angle
# plt.plot(x, planeAngle(x, 0, k, w))
# plt.show()


###### Superposition
# plt.plot(x,0.5*np.real(planeWave(x, 0, k, w))+0.5*np.real(planeWave(x, 0, 2*k, w)))
# plt.show()

global time
time = 0
fig, ax = plt.subplots()
real, = ax.plot(x, np.real(planeWave(x, time, k, w)))

imag, = ax.plot(x, np.imag(planeWave(x, time, k, w)))


def update(i):
    real.set_data(x, np.real(planeWave(x, time, k, w)))
    imag.set_data(x, np.imag(planeWave(x, time, k, w)))
    time+=dt
    return real, imag,


anim = animation.FuncAnimation(fig, update, interval=1, blit=True)
plt.show()
