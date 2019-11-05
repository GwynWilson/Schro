import numpy as np
N = 2 ** 12
dx = 0.1
x = np.array([i * dx for i in range(N)])

hbar = 1
m = 1
scale = 1

k0 = 2
x0 = int(N / 4) * dx
sig = 8

bar_amp = 1
omeg = 10
x1 = int(N / 2) * dx

dt = 0.01
step = 50
t_final = 110