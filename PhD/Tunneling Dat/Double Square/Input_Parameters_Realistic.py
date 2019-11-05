import numpy as np

m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
scale = 2 * np.pi * hbar

lim = 2 * 10 ** - 5
N = 2 ** 11
# N = 2 ** 6
dx = 2 * lim / N

dt = 10 ** -7
step = 1000
t_final = 0.007
dk = 2 * np.pi / (N * dx)

k_lim = np.pi / dx
k1 = -k_lim + dk * np.arange(N)

# x = np.array([i * dx for i in range(N)])
x = np.arange(-lim, lim, dx)

bar_amp = scale * 30 * 10 ** 3
wid = (1 / 3) * 10 ** -6
sep = (5 / 8) * 10 ** -6
x1 = 0

omeg = 10**-6
separation = 5*10**-6

sig = 1 * 10 ** -6
E = scale * 10 ** 3
k0 = np.sqrt(2 * m * E / (hbar ** 2) - 1 / (4 * sig ** 2))
x0 = -1 * 10 ** -5

if __name__ == "__main__":
    print(k0, k_lim)
    from Numerical_Constants import Constants

    Constants(bar_amp, dt, dx, k0)
