import Ito_Process as ito
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def a(x, t):
    return 0


def b(x, t):
    return amp


def modBs(B1, B2, I1, I2, theta):
    return np.sqrt((B1 * I1) ** 2 + (B2 * I2) ** 2 + 2 * B1 * B2 * I1 * I2 * np.cos(theta))


def Bu(B1, B2, I1, I2, theta):
    return np.array([B1 * I1 + B2 * I2 * np.cos(theta), B2 * I2 * np.sin(theta)]) / modBs(B1, B2, I1, I2, theta)


def Bo(B1, B2, I1, I2, theta):
    u = Bu(B1, B2, I1, I2, theta)
    u[1] = -u[1]
    return u


def sigU(B, I, theta, sig):
    return np.sqrt(2) * B ** 2 * I * (1 + np.cos(theta)) * sig / modBs(B, B, I, I, theta)


def sigO(B, I, theta, sig):
    return np.sqrt(2) * B ** 2 * I * np.sin(theta) * sig / modBs(B, B, I, I, theta)


def phi(B1, B2, I1, I2, theta):
    return np.arctan(B2 * I2 * np.sin(theta) / (B1 * I1 + B2 * I2 * np.cos(theta)))


def oneRun(n, dt, a, b, x0, B1_mag, B2_mag, thet):
    run1 = ito.itoProcess(n, dt, a, b, x0=x0)
    B1_x = B1_mag * np.asarray(run1)
    B1_y = np.zeros(n)

    run2 = ito.itoProcess(n, dt, a, b, x0=x0)
    B2_x = B2_mag * np.asarray(run2) * np.cos(thet)
    B2_y = B2_mag * np.asarray(run2) * np.sin(thet)
    return B1_x + B2_x, B1_y + B2_y


def plotElipse(x_pos, y_pos, width, height, t_rot, label=None, color=None):
    n = 1000
    ang = np.linspace(0, 2 * np.pi, n)
    elipse = np.array([width * np.cos(ang), height * np.sin(ang)])
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])
    elipse_rot = np.zeros((2, n))
    for i in range(n):
        elipse_rot[:, i] = np.dot(R_rot, elipse[:, i])
    # plt.plot(Av_x + width * np.cos(ang), Av_y + height * np.sin(ang))
    plt.plot(x_pos + elipse_rot[0, :], y_pos + elipse_rot[1, :], label=label, color=color)


I = 1
amp = 10 ** (-5)

time = 1
dt = 10 ** -4
n = int(time / dt)

Bmag = 10
thet = np.pi / 4

Av_x = Bmag * I + Bmag * I * np.cos(thet)
Av_y = Bmag * I * np.sin(thet)

Unit = Bu(Bmag, Bmag, I, I, thet)
Orth = Bo(Bmag, Bmag, I, I, thet)
phi_av = phi(Bmag, Bmag, I, I, thet)

########## Unit Vecs
# print(Unit)
# print(Orth)
# plt.scatter(Unit[0], Unit[1])
# plt.scatter(Orth[0], Orth[1])
# plt.scatter(np.cos(phi_av), np.sin(phi_av))
# plt.show()

######### Weiner Process
# nRuns = 10
# for i in range(nRuns):
#     runx, runy = oneRun(n, dt, a, b, I, Bmag, Bmag, thet)
#     plt.scatter(runx, runy)
# plt.scatter(Av_x,Av_y)
# plt.show()

####### White Gaussian
B1_x = Bmag * (ito.randGauss(I, amp, n))
B1_y = np.zeros(n)
B2_n = Bmag * (ito.randGauss(I, amp, n))
B2_x = B2_n * np.cos(thet)
B2_y = B2_n * np.sin(thet)
Bs_x = B1_x + B2_x
Bs_y = B1_y + B2_y

width = sigU(Bmag, I, thet, amp)
height = sigO(Bmag, I, thet, amp)

origin = [0], [0]
fig,ax = plt.subplots()
plt.title("Magnetic field Vector Addition")
ax.set_aspect("equal")
plt.quiver(*origin, [Av_x], [Av_y], units="xy", scale=1)
plt.quiver(*origin, [Bmag*I], [0], units="xy", scale=1)
plt.quiver(*origin, [Bmag *I* np.cos(thet)], [Bmag*I * np.sin(thet)], units="xy", scale=1)
plt.scatter(Bs_x, Bs_y, s=5, marker=".", color="c")
plt.xlim((0, 2 * Bmag))
plt.ylim((0, Bmag))
plt.show()

plt.figure()
plt.scatter(Bs_x, Bs_y, s=5, marker=".", color="c")
plotElipse(Av_x, Av_y, width, height, phi_av, label="1 Sig")
plotElipse(Av_x, Av_y, 2 * width, 2 * height, phi_av, label="2 Sig", color="k")
plotElipse(Av_x, Av_y, 3 * width, 3 * height, phi_av, label="3 Sig")
plt.xlim((Av_x - 4 * width, Av_x + 4 * width))
plt.ylim((Av_y - 4 * height, Av_y + 4 * height))
plt.legend()
plt.autoscale(False)
plt.title(f"Sum of noisy B fields B={Bmag},theta={round(thet, 3)},noise={amp}")
plt.show()
