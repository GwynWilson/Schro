import numpy as np
import matplotlib.pyplot as plt


def Bz(y, z):
    return mu0 * I / (4 * np.pi) * (((z - d / 2) ** 2 + y ** 2) ** (-1) + ((z + d / 2) ** 2 + y ** 2) ** (-1))


def BzI(y, z, I):
    return mu0 * I / (4 * np.pi) * (((z - d / 2) ** 2 + y ** 2) ** (-1) + ((z + d / 2) ** 2 + y ** 2) ** (-1))


def By(y):
    return (mu0 * I / (2 * np.pi * y)) * np.arctan(d / (2 * y))


def B1(x, y, z):
    if x <= 0:
        return mu0 * I / (4 * np.pi) * ((z + d / 2) ** 2 + y ** 2) ** (-1 / 2)
    else:
        return 0


def B3(x, y, z):
    if x >= 0:
        return mu0 * I / (4 * np.pi) * ((z - d / 2) ** 2 + y ** 2) ** (-1 / 2)
    else:
        return 0


def B2(x, y, z):
    if z <= -d / 2 or z >= d / 2:
        return 0
    else:
        r = np.sqrt(x ** 2 + y ** 2)
        return mu0 * I / (4 * np.pi * r) * (np.sin(np.arctan(((d / 2) + z) / r)) + np.sin(np.arctan(((d / 2) - z) / r)))


mu0 = 1
I = 2
Bx = 10
y0 = mu0 * I / (2 * np.pi * Bx)
d = 2

mu0 = 8.85418782 * 10 ** (-12)
I = 2
Bx = 10 ** (-3)
y0 = mu0 * I / (2 * np.pi * Bx)
d = 50 * 10 ** -6

y_list = np.linspace(y0 / 2, 5 * y0, 100)
y_list = np.linspace(0.2, 1.2, 100) * 10 ** (-8)
z_list = np.linspace(-d / 4, d / 4, 100)
I_list = np.linspace(I, 2 * I, 100)

######Wire Trap
# plt.plot(y_list, abs(By(y_list) - Bx))
# plt.axvline(y0)
# plt.title("By for minimum value of y0")
# plt.xlabel("y")
# plt.ylabel("By(y)")
# plt.savefig("By")
# plt.show()

#### Axial Trap
# plt.plot(z_list, Bz(y0, z_list))
# plt.title("Bz for minimum value of y0")
# plt.xlabel("z")
# plt.ylabel("Bz(z)")
# plt.savefig("Bz")
# plt.show()

######### Total field
# Z, Y = np.meshgrid(z_list, y_list)
# Bz_list = Bz(Y, Z)
# By_list = By(Y)
# Btot = abs(By(Y) - Bx) + Bz(Y, Z)
# a = plt.contourf(Z, Y, Btot, 20)
# plt.colorbar(a)
# plt.title("Total magnetic field in x=0 plane")
# plt.xlabel("z")
# plt.ylabel("y")
# plt.savefig("Btot")
# plt.show()

########### Current vs Field strength
# X, Y = np.meshgrid(I_list, z_list)
# Z = BzI(y0, Y, X)
# a = plt.contourf(X, Y, Z)
# plt.colorbar(a)
# plt.show()
