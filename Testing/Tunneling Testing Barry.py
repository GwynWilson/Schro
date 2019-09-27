import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, fftshift
from Schrodinger_Solver import Schrodinger


def gauss_init(x, k0, x0=0, d=1):
    """
    Initial wave function definition
    :param x: X points
    :param k0: Initial momentum
    :param x0: Position of mean
    :param d: Width of wave packet
    :return: Returns array containing initial wave function
    """
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (d ** 2)) * np.exp(1j * k0 * x)


def barrier(x, A, x1, x2):
    """
    Makes potential barrier
    :param x: x points
    :param A: Barrier amplitude (v0)
    :param x1: left end of barrier
    :param x2: right point of barrier
    :return: Returns array containing barrier
    """
    temp = np.zeros(len(x))
    for n, v in enumerate(x):
        if v < x1:
            temp[n] = 0
        elif v > x2:
            temp[n] = 0
        else:
            temp[n] = A

    return temp


def run(x2, time=False):
    """
    Runs the tunneling simulation for a given barrier length.
    :param x2: Second barrier point
    :param time: Boolean, can be set to tre to return the time increments
    :return: returns the tunneling probability as a function of time
    """
    # Initialing for a new run
    V_x = barrier(x, A, x1, x2)
    psi_init = gauss_init(x, k_init, x0=x0, d=sig)
    sch = Schrodinger(x, psi_init, V_x, k, hbar=hbar, m=m, t=0, args=x2)

    dat = []
    t_list = []

    # Loops runs the sim
    for i in range(0, Ns):
        sch.evolve_t(step, dt)
        dat.append(sch.barrier_transmition())  # Appends tunneling probability
        t_list.append(sch.t)  # Appending time

    if time:
        return dat, t_list
    else:
        return dat


def Theory_Full(L, V0, E, m=1, hbar=1):
    """
    Exact Tunneling probability for plane waves
    :param L: Barrier Length
    :param V0: Barrier Height
    :param E: Energy of wave packet
    :param m: Mass of Packet
    :param hbar: h bar
    :return: Returns value for tunneling probability
    """
    k1 = np.sqrt(((2 * m * E) / hbar ** 2))
    k2 = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return (1 + 1 / 4 * (k1 / k2 + k2 / k1) ** 2 * np.sinh(k2 * L) ** 2) ** (-1)


def Theory_Exp(L, V0, E, m=1, hbar=1):
    """
    Approximation of Tunneling probability for plane waves. Valid for low tunneling probability
    :param L: Barrier Length
    :param V0: Barrier Height
    :param E: Energy of wave packet
    :param m: Mass of Packet
    :param hbar: h bar
    :return: Returns value for tunneling probability
    """
    amp = 16 * E / V0 * (1 - (E / V0))
    coeff = np.sqrt(((2 * m) / (hbar ** 2)) * (V0 - E))
    return amp * np.exp(-2 * L * coeff)


# Defining x axis
N = 2 ** 12
dx = 0.02
x_length = N * dx

x = np.zeros(N)
for i in range(0, N):
    x[i] = i * dx

# Misc Variables
hbar = 1
m = 1

# Barrier Definitions
A = 60
x1 = int(0.5 * N) * dx  # Left side of Barrier
L = 10 * dx  # Length
x2 = x1 + L
V_x = barrier(x, A, x1, x2)

# Wave Function definitions
x0 = int(0.3 * x_length)
sig = 6
k_init = 2
psi_init = gauss_init(x, k_init, x0=x0, d=sig)
E = (hbar ** 2) * (k_init ** 2) / (2 * m)

# Defining K Space
dk = dx / (2 * np.pi)
k = fftfreq(N, dk)
ks = fftshift(k)

# Time definitions
t = 0
dt = 0.001
step = 50  # Number of steps each evolution
Ns = 380  # Total number of evolutions (Chosen such that the wave packet fully tunnels through)

# Constants
print("Final time : ", dt * Ns * step)
print("Barrier Length", L)
print("Diffusion", dt / (dx ** 2))
print("vdt", A * dt)
print("kdx", k_init * dx)

tprob, time = run(x2, time=True)  # Runs simulation

plt.plot(time, tprob)
plt.title("Tunneled Packet vs Time")
plt.xlabel("Time")
plt.ylabel("Probability of Particle Tunnelling")
plt.savefig("Tunnel_Probability.png")
plt.show()

print("Simulated Tunneling Probability", tprob[-1])  # Final tunneling time
print("Theoretical Tunneling Probability", Theory_Full(L, A, E, m=m, hbar=hbar))
print("Exponential Tunneling Probability", Theory_Exp(L, A, E, m=m, hbar=hbar))
