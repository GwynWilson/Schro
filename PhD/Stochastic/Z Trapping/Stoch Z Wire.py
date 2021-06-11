import numpy as np
import matplotlib.pyplot as plt


def y0FirstOrder(mu0, Bx, I, d):
    return mu0 * I / (2 * np.pi * Bx)

def y0Full(mu0, Bx, I, d):
    k = mu0 * I / (2 * np.pi)
    return (d/2 *(d**2 + 4*(k/Bx)**2)**(1/2) - d**2/2)**(1/2)


mu0 = 1.2566370614 * 10 ** (-6)
I = 300 * 10 ** (-3)
Bx = 9 * 10 ** (-5)
y0 = mu0 * I / (2 * np.pi * Bx)
y1_aprox = mu0 * I / (2 * np.pi * Bx)
d = 1.9 * 10 ** (-3)
sig = 1 * 10 ** (-6)
k = mu0 * I / (2 * np.pi)

print(d**2,k**2/Bx**2)


N = 10000
Il = I + sig * np.random.randn(N)
print(np.mean(Il))
print(np.sqrt(np.var(Il)))

centre = y0Full(mu0, Bx, Il, d)
print(np.mean(centre))
print(np.sqrt(np.var(centre)))


plt.plot(range(N), y0Full(mu0, Bx, Il, d))
plt.show()

