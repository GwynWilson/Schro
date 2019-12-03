import numpy as np

m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
scale = 2 * np.pi * hbar

lim = 4 * 10 ** - 5
N = 2 ** 12
dx = 2 * lim / N

dt = 10 ** -6
step = 50
t_final = 0.007
dk = 2 * np.pi / (N * dx)

k_lim = np.pi / dx
k1 = -k_lim + dk * np.arange(N)

# x = np.array([i * dx for i in range(N)])
x = np.arange(-lim, lim, dx)

x0 = -2 * 10 ** -5
x1 = x[int(N / 2)]

sig = 1 * 10 ** -6
E = scale * 10 ** 3
k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))

bar_amp = scale * 10 ** 3
w = 10 ** -6
L = 4*10**-6


if __name__=="__main__":
    from Numerical_Constants import Constants
    Constants(bar_amp,dt,dx,k0)