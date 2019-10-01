from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (d ** 2)) * np.exp(1j * k0 * x)


N = 2 ** 11
dx = 0.1
x = np.array([i * dx for i in range(N)])
x = x - max(x)/2

hbar = 1
m = 1

k0 = 0
x0 = x[int(N/2)]
sig = 2

Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = np.zeros(N)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m)

E = (hbar ** 2) * (k0 ** 2) / (2 * m)

print("Theoretical Energy", E)
print("Simulated E", sch.energy())

plt.plot(x,sch.mod_square_x(r=True))
plt.show()