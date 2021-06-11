import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.integrate import simps


def detectFolder():
    import os
    cwd = os.getcwd()
    folder = "Dat"
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except OSError:
            print("whoops")


def genNoise(n, dt):
    """
    Will generate array of length 2n of gaussian white noise
    :param n:
    :return:
    """
    return np.sqrt(dt) * np.random.randn(2 * n)


def f(t, x, v):
    return v


def g(t, x, v, a):
    return -a * x


def h(t, x, v, a, m):
    return 0.5 * m * a * (x ** 2) + 0.5 * m * (v ** 2)


def rk4StocasticPhase(variables, noise=None):
    n, dt, x0, v0, w, sig, m = variables
    a = w ** 2
    b = sig / m
    xl = [x0]
    vl = [v0]
    pl = [0]
    varl = [0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    p = 0
    var = 0
    if not isinstance(noise, np.ndarray):
        noise = genNoise(n, dt)
    for i in range(n - 1):
        k0 = dt * f(t, x, v)
        l0 = dt * g(t, x, v, a) + b * noise[2 * i]
        j0 = dt * h(t, x, v, a, m)

        k1 = dt * f(t + dt / 2, x + k0 / 2, v + l0 / 2)
        l1 = dt * g(t + dt / 2, x + k0 / 2, v + l0 / 2, a) + b * noise[2 * i]
        j1 = dt * h(t + dt / 2, x + k0 / 2, v + l0 / 2, a, m)

        k2 = dt * f(t + dt / 2, x + k1 / 2, v + l1 / 2)
        l2 = dt * g(t + dt / 2, x + k1 / 2, v + l1 / 2, a) + b * noise[2 * i]
        j2 = dt * h(t + dt / 2, x + k1 / 2, v + l1 / 2, a, m)

        k3 = dt * f(t + dt, x + k2, v + l2)
        l3 = dt * g(t + dt, x + k2, v + l2, a) + b * noise[2 * i + 2]
        j3 = dt * h(t + dt, x + k2, v + l2, a, m)

        x += (k0 + 2 * k1 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6
        p += (j0 + 2 * j1 + 2 * j2 + j3) / 6
        var += 2 * dt * h(t, x, v, a, m) * p
        t += dt

        xl.append(x)
        vl.append(v)
        tl.append(t)
        pl.append(p)
        varl.append(var)

    return tl, np.asarray(xl), np.asarray(vl), np.asarray(pl), np.asarray(varl)


def dtRun(n_list, variables, average=1):
    t, x0, v0, w, sig, m = variables
    phase_final = []
    for ind, i in enumerate(n_list):
        if ind % 100 == 0:
            print(ind)
        dti = t / i
        tempvar = [i, dti, x0, v0, w, sig, m]
        temp = 0
        for j in range(average):
            tl, xl, vl, pl, vrl = rk4StocasticPhase(tempvar)
            temp += pl[-1]
        phase_final.append(temp / average)
    return phase_final


def dtDat(n_list, variables, add="", average=1):
    points = int(len(n_list))
    if add != "":
        add = "_" + str(add)
    phase_final = dtRun(n_list, variables, average=average)
    np.savez_compressed(f"Dat/dt_{points}{add}", n_list=n_list, phase_final=phase_final)
    np.savez(f"Dat/dt_{points}_variables{add}", variables)
    return 0


def averagePhaseTheory(variables):
    T, x0, v0, w, sig, m = variables
    return (m * w ** 2 * x0 ** 2 / (2 * hbar) + m * v0 ** 2 / (2 * hbar)) * T + sig ** 2 * T ** 2 / (4 * m * hbar)


def loadDtDat(points, add=""):
    if add != "":
        add = "_" + add

    dat = np.load(f"Dat/dt_{points}_variables{add}.npz")
    t, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [t, x0, v0, w, sig, m]

    dat = np.load(f"Dat/dt_{points}{add}.npz")
    n_list = dat["n_list"]
    dt_list = [t / i for i in n_list]
    phase_final = dat["phase_final"] / hbar

    expect = averagePhaseTheory(var_list)

    sample = 3
    plt.plot(dt_list[::sample], phase_final[::sample], linestyle="none", marker="o", label="Simulation")
    plt.plot((dt_list[0], dt_list[-1]), (expect, expect), label="Theoretical")
    plt.title("dt Vs Final Phase")
    plt.xlabel("dt")
    plt.ylabel("Phase")
    plt.legend()
    plt.show()


def averageRun(av_list, variables):
    phase_final = []
    for ind, i in enumerate(av_list):
        if ind % 10 == 0:
            print(ind)
        temp = 0
        for j in range(i):
            tl, xl, vl, pl, vrl = rk4StocasticPhase(variables)
            temp += pl[-1]
        phase_final.append(temp / i)
    return phase_final


def averageDat(n_list, variables, add=""):
    points = int(len(n_list))
    if add != "":
        add = "_" + str(add)
    phase_final = averageRun(n_list, variables)
    np.savez_compressed(f"Dat/av_{points}{add}", n_list=n_list, phase_final=phase_final)
    np.savez(f"Dat/av_{points}_variables{add}", variables)
    return 0


def loadAverageDat(points, add=""):
    if add != "":
        add = "_" + add

    dat = np.load(f"Dat/av_{points}_variables{add}.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [t, x0, v0, w, sig, m]

    dat = np.load(f"Dat/av_{points}{add}.npz")
    n_list = dat["n_list"]
    phase_final = dat["phase_final"] / hbar

    expect = averagePhaseTheory(var_list)

    plt.semilogx(n_list, phase_final, linestyle="none", marker="o", label="Simulation")
    plt.plot((n_list[0], n_list[-1]), (expect, expect), label="Theoretical")
    plt.title("Number of Runs Vs Final Phase")
    plt.xlabel("Runs")
    plt.ylabel("Phase")
    plt.legend()
    plt.show()


d = 0.004
v = 0.01
t = d / v
print("Final time", t)
n = 500
dt = t / n

m = 1.44 * 10 ** (-25)
w = 6.1 * 10 ** 2
k = w ** 2 * m

print("t0", w ** (-1))
hbar = 1.0545718 * 10 ** -34

sig = m / 100

x0 = 0
v0 = 0

var_list = [t, x0, v0, w, sig, m]

################## dt Runs
n_min = 200
n_max = 10000
points = 5000

dt_list = np.linspace(t / n_min, t / n_max, points)
n_list = t / dt_list
n_list = n_list.astype("int")

average = 100
# dtDat(n_list, var_list,add=f"av_{average}",average=average)
loadDtDat(points, add=f"av_{average}")

################### Average Runs

dt = 5e-4
n = int(t / dt)
points = 100
av_list = np.linspace(500, 5000, points, dtype=int)

var_list = [n, dt, x0, v0, w, sig, m]
# averageDat(av_list, var_list)
loadAverageDat(points)
