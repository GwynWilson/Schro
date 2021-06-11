import numpy as np
import matplotlib.pyplot as plt


def gauss(x, A, sig):
    return A * np.exp(-(x / sig) ** 2)


def square(x, A, L):
    temp = []
    for i in x:
        if i < -L / 2 or i > L / 2:
            temp.append(0)
        else:
            temp.append(A)
    return np.asarray(temp)


def Impedence(v, E, m=1, hbar=1, dx=1):
    for n, i in enumerate(reversed(v)):
        diff = (E - i)
        if diff == 0:
            diff += 10 ** -99
        K = 1j * np.sqrt(2 * m * diff + 0j) / hbar
        z0 = -1j * hbar * K / m
        if n == 0:
            zload = z0
        else:
            zload = zin
        zin = z0 * ((zload * np.cosh(K * dx) - z0 * np.sinh(K * dx)) / (
                z0 * np.cosh(K * dx) - zload * np.sinh(K * dx)))

    coeff = np.real(((zin - z0) / (zin + z0)) * np.conj((zin - z0) / (zin + z0)))
    return 1 - coeff


m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
V0 = 10 ** -30

e = V0
l = np.sqrt(hbar ** 2 / (2 * m * V0))
L = 10 * l

x = np.linspace(-10, 10, 1000) * l
dx = x[1] - x[0]

# m = 10
# hbar = 10
# L = 0.7
# V0 = 10
#
# e = V0
# l = np.sqrt(hbar ** 2 / (2 * m * V0))
# print(l)
#
# x = np.linspace(-2, 2, 100)*L
# dx = x[1] - x[0]

bar = square(x, V0, L)
# V0_scale = V0 / e
# L_scale = L / l
# bar_scale = square(x/l,V0_scale,L_scale)

#
# plt.plot(x, bar)
# plt.show()

# plt.plot(x / l, bar/e)
# plt.show()

plt.rcParams.update({"font.size": 14})
L_list = np.array([1, 5, 10]) * l
# for L_i in L_list:
#     plt.plot(x / l, square(x, V0, L_i) / V0, label=f"{L_i/l:.0f}")
# plt.legend()
# plt.xlabel(r"$x/l$")
# plt.ylabel(r"$V(x)/\epsilon$")
# # plt.title("Square barrier varying width")
# plt.savefig("Scaled Width")
# plt.show()

E_list = np.linspace(0.1, 5, 500) * V0
for L_i in L_list:
    bar = square(x, V0, L_i)
    T_prob1 = []
    for E in E_list:
        imp = Impedence(bar, E, m=m, hbar=hbar, dx=dx)
        T_prob1.append(imp)
    Temp = np.insert(E_list, 0, 0)
    T_prob1 = [0] + T_prob1
    plt.plot(Temp / V0, T_prob1, label=f"{L_i/l:.0f}")

plt.legend()
plt.xlabel(r"$E/V_0$")
plt.ylabel("Transmission Probability")
# plt.title("Varying Width Transmission")
plt.savefig("Width Transmission")
plt.show()
