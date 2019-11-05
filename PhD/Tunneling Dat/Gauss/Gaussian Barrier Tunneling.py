from Schrodinger_Solver import Schrodinger
from Animate import Animate
from Numerical_Constants import Constants
import numpy as np
import matplotlib.pyplot as plt

from Input_Parameters_Realistic import *


def gauss_init(x, k0, x0=0, d=1):
    return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * (d ** 2))) * np.exp(1j * k0 * x)


def gauss_barrier(x, A, x0, w):
    return A * (np.exp(-(x - x0) ** 2 / w ** 2))


def run_A(A):
    Psi_x = gauss_init(x, k0, x0=x0, d=sig)
    V_x = gauss_barrier(x, A, x1, omeg)
    sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0, args=x1)

    imp = sch.impedencePacket(tol=10**-11)
    time_list = []
    Trans_list = []
    while sch.t < t_final:
        sch.evolve_t(N_steps=step, dt=dt)
        time_list.append(sch.t)
        Trans_list.append(sch.barrier_transmition())
        print("Height {v}, Time {t}".format(v=A/scale, t=sch.t))

    return sch.barrier_transmition(), imp


Psi_x = gauss_init(x, k0, x0=x0, d=sig)
V_x = gauss_barrier(x, bar_amp, x1, omeg)

sch = Schrodinger(x, Psi_x, V_x, hbar=hbar, m=m, t=0, args=x1)
print("Energy", sch.energy())
print("Imp",sch.impedencePacket(tol=10**-11))

# plt.plot(sch.k, sch.mod_square_k(r=True))
# plt.plot(x, np.asarray(V_x)/scale)
# plt.show()

# a = Animate(sch, V_x, step, dt, lim1=((x[0], x[-1]), (0, max(np.real(sch.psi_squared)))),
#             lim2=((sch.k[0], sch.k[-1]), (0, max(np.real(sch.psi_k)))))
# a.make_fig()

# v_list = np.arange(0.5, 5, 0.1)


v_list = np.linspace(0.1, 3, 50) * bar_amp
T_list = []
I_list = []
for i in v_list:
    T, I = run_A(i)
    T_list.append(T)
    I_list.append(I)

save = [0 for k in range(len(v_list))]
for i in range(len(v_list)):
    save[i] = (v_list[i], T_list[i], I_list[i])

var = [N, dx, omeg, dt, k0]

np.savetxt("Gauss_Barrier_Real.txt", save)
np.savetxt("Gauss_Barrier_var_Real.txt", var)

plt.plot(v_list, T_list)
plt.show()
