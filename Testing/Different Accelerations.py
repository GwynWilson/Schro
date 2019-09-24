import numpy as np
import matplotlib.pyplot as plt


def xpos(t, xi, omega, a, x0):
    w = omega
    return (xi + (a * w ** 2)) * np.cos(w * t) + (0.5 * a * (t ** 2)) + x0 - (a * (w ** 2)) - 0.5 * a * t ** 2


def xpos2(t, xi, omega, a, x0):
    w = omega
    return (xi + (a * w ** 2)) * np.cos(w * t) + (0.5 * a * (t ** 2)) + x0 - (a * (w ** 2))


def motion(t, x0, a):
    return x0 + 0.5 * a * t ** 2


x0 = 10
xi = -10
A = x0 + xi

m = 1
omega = 6
a = 0.1
args = (m, omega, x0, a)

t_list = np.linspace(0, 10, 1000)
a_list = np.linspace(0.01, 0.1, 3)
# a_list = [0.01, 0.1, 1]
# x_list = [xpos(j, xi, omega, a, x0) for j in t_list]

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.set_title("Motion for Differing Accelerations")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Position")
for a in a_list:
    print(a)
    temp = [xpos(j, xi, omega, a, x0) for j in t_list]
    ax.plot(t_list, temp, label='a={v}'.format(v=round(a, 3)))
    # ax.plot(t_list, motion(t_list, x0,a), linestyle="--")

ax.legend()
plt.savefig("accelerating_osc_omeg")
plt.show()

"""
x_list1 = [xpos(j, xi, omega, 0.01, x0) for j in t_list]
x_list2 = [xpos(j, xi, omega, 0.02, x0) for j in t_list]
x_list3 = [xpos(j, xi, omega, 0.03, x0) for j in t_list]


plt.plot(t_list, x_list1)
plt.plot(t_list, x_list2)
plt.plot(t_list, x_list3)
plt.show()
"""
