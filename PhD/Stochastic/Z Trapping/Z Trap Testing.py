import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def B1(x, y, z):
    return mu0 * I / (4 * np.pi) * (1 / (np.sqrt((d + z) ** 2 + y ** 2)))


def B2(x, y, z):
    return mu0 * I / (4 * np.pi * y) * (
            (d + z) / (np.sqrt((d + z) ** 2 + y ** 2)) + (d - z) / (np.sqrt((d - z) ** 2 + y ** 2)))


def B3(x, y, z):
    return mu0 * I / (4 * np.pi) * (1 / (np.sqrt((d - z) ** 2 + y ** 2)))


def totB(x, y, z, Btilde):
    temp = np.zeros(3)
    temp[0] = Btilde - B2(x, y, z)
    temp[1] = B3(x, y, z) * (d - z) / (np.sqrt((d - z) ** 2 + y ** 2)) - B1(x, y, z) * (d + z) / (
        np.sqrt((d + z) ** 2 + y ** 2))
    temp[2] = B1(x, y, z) * y / (np.sqrt((d + z) ** 2 + y ** 2)) + B3(x, y, z) * y / (np.sqrt((d - z) ** 2 + y ** 2))
    return temp


def magB(x, y, z, Btilde):
    vec = totB(x, y, z, Btilde)
    mag = 0
    for i in vec:
        mag += i ** 2
    return np.sqrt(mag)


def T_array(X_list, Y_list):
    temp = np.zeros((len(Y_list), len(X_list)))
    for i, yv in enumerate(Y_list):
        # print(f"{i}")
        for j, xv in enumerate(X_list):
            temp[i, j] = magB(0, yv, xv, Bx)
    return temp


def wire(r):
    return mu0 * I / (2 * np.pi * r)


def root(y):
    return Bx - mu0 * I / (2 * np.pi * y) * (d / np.sqrt(d ** 2 + y ** 2))


mu0 = 1
I = 2
Bx = 10
y0 = mu0 * I / (2 * np.pi * Bx)
d = 2

mu0 = 1.2566370614 * 10 ** (-6)
I = 300 * 10 ** (-3)
Bx = 9 * 10 ** (-5)
y0 = mu0 * I / (2 * np.pi * Bx)
y1_aprox = mu0 * I / (2 * np.pi * Bx)
d = 1.9 * 10 ** (-3)

k = mu0 * I / (2 * np.pi)

n = 100
y_list = np.linspace(0.5 * y0, 5 * y0, n)
z_list = np.linspace(-d, d, n)

y_zoom = np.linspace(0.95, 1.05, 10000) * y0
B_centre = abs(Bx - B2(0, y_zoom, 0))
y1_ind = np.where(B_centre == min(B_centre))[0][0]
y1 = y_zoom[y1_ind]
print("Actual min", y1)
print("Aprox Min", k / Bx * (1 - k ** 2 / (2 * Bx ** 2 * d ** 2)))
print("y0", y0)
print(y0**3/d**3)

####Wire
# plt.plot(y_list, abs(Bx - wire(y_list)), label="Infinite")
# plt.plot(y_list, abs(Bx - B2(0, y_list, 0)), linestyle="--", label="Finite")
# plt.axvline(y0, linestyle=":", label="y0")
# plt.title("Vertical confinement due to central wire")
# plt.xlabel("y (m)")
# plt.ylabel("|B| (Kg/As^2)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("Confinement due to central wire")
# plt.show()


# plt.plot(y_zoom/10**3, abs(Bx - wire(y_zoom)), label="Infinite")
# plt.plot(y_zoom/10**3, abs(Bx - B2(0, y_zoom, 0)), linestyle="--", label="Finite")
# plt.axvline(y0/10**3, linestyle=":", label="y0")
# plt.title("Minimum of B field")
# plt.xlabel("y (mm)")
# plt.ylabel("|B| (Kg/As^2)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("Minimum of B field")
# plt.show()

###### All axies
# B_x = [totB(0, y1, i, Bx)[0] for i in z_list]
# B_y = [totB(0, y1, i, Bx)[1] for i in z_list]
# B_z = [totB(0, y1, i, Bx)[2] for i in z_list]
# B_total = [magB(0, y1, i, Bx) for i in z_list]
# scale = 10**-3
#
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
# fig.suptitle(f"Magnetic field of each component, Btilde={Bx}")
# axs[0, 0].plot(z_list/scale, B_x)
# axs[0, 0].set_title("B_x")
# axs[0, 1].plot(z_list/scale, B_y)
# axs[0, 1].set_title("B_y")
# axs[1, 0].plot(z_list/scale, B_z)
# axs[1, 0].set_title("B_z")
# axs[1, 1].plot(z_list/scale, B_total)
# axs[1, 1].set_title("|B|")
#
# fig.text(0.5, 0.05, "Z (mm)", va='center', ha='center')
# fig.text(0.04, 0.5, "B (Kg/As^2)", va='center', ha='center', rotation='vertical')
#
# # plt.savefig(f"All_Fields_{Bx}")
# plt.show()

#####Contour
# y_list = np.linspace(0.5 * y0, 3 * y0, n)
# Z, Y = np.meshgrid(z_list, y_list)
# Btot = T_array(z_list, y_list)
# a = plt.contourf(Z, Y, Btot, 20, cmap="gist_heat")
# plt.colorbar(a)
# plt.title("Total magnetic field in x=0 plane")
# plt.xlabel("z")
# plt.ylabel("y")
# plt.tight_layout()
# plt.savefig(f"Btot_cont_{Bx}")
# plt.show()

###### Roots
# plt.plot(y_list, root(y_list))
# plt.show()
# ysolved = fsolve(root, x0=y0)[0]
# print(f"Z trap minima y={ysolved} \nInfinite wire minima y={y0} \nDiffernce = {abs(ysolved-y0)}")


###### Harmonic Trapping
# z_zoom = np.linspace(-d / 4, d / 4, n)
# B_z_y1 = [totB(0, y1, i, Bx) for i in z_zoom]
# B_z = [B_z_y1[i][2] for i in range(n)]
#
# plt.plot(z_list, B_z)
# plt.show()


#### Wire x direc
# r_list = np.sqrt(np.asarray(z_list)**2+y0**2)
# plt.plot(Bx-wire(r_list))
# plt.show()
