import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import time
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


def rk4Stocastic(n, dt, x0, v0, a, b, noise=None):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = v0
    if not isinstance(noise, np.ndarray):
        noise = genNoise(n, dt)
    for i in range(n - 1):
        k0 = dt * f(t, x, v)
        l0 = dt * g(t, x, v, a) + b * noise[2 * i]
        k1 = dt * f(t + dt / 2, x + k0 / 2, v + l0 / 2)
        l1 = dt * g(t + dt / 2, x + k0 / 2, v + l0 / 2, a) + b * noise[2 * i]
        k2 = dt * f(t + dt / 2, x + k1 / 2, v + l1 / 2)
        l2 = dt * g(t + dt / 2, x + k1 / 2, v + l1 / 2, a) + b * noise[2 * i]
        k3 = dt * f(t + dt, x + k2, v + l2)
        l3 = dt * g(t + dt, x + k2, v + l2, a) + b * noise[2 * i + 2]

        x += (k0 + 2 * k1 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6
        t += dt

        xl.append(x)
        vl.append(v)
        tl.append(t)

    return tl, np.asarray(xl), np.asarray(vl)


def xHarmonic(x0, v0, w, t):
    t = np.asarray(t)
    return x0 * np.cos(w * t) + v0 * np.sin(w * t) / w


def vHarmonic(x0, v0, w, t):
    t = np.asarray(t)
    return -w * x0 * np.sin(w * t) + v0 * np.cos(w * t)


def xVarHarmonic(sig, m, w, t):
    t = np.asarray(t)
    return sig ** 2 / (2 * (w ** 2) * (m ** 2)) * (t - np.sin(2 * w * t) / (2 * w))


def vVarHarmonic(sig, m, w, t):
    t = np.asarray(t)
    return (sig ** 2 / (2 * (m ** 2))) * (t + np.sin(2 * w * t) / (2 * w))


def averageRuns(n_runs, variables, method):
    n, dt, x0, v0, w, sig, m = variables
    x_sum = np.zeros(n)
    x_square = np.zeros(n)
    v_sum = np.zeros(n)
    v_square = np.zeros(n)
    xv = np.zeros(n)
    prev = time.time()
    for n_run in range(n_runs):
        if n_run % 1000 == 0:
            # now = time.time()
            # diff = now - prev
            # d_left = diff * (n_runs - n_run)
            # hr = d_left/3600
            # min = (hr-np.floor(hr))*60
            # sec = (min-np.floor(min))*60
            # print(f"Run: {n_run}\nTime: {diff}\nLeft: {int(np.floor(hr))}:{int(np.floor(min))}:{int(np.floor(sec))}")
            # prev = now
            print(n_run)
        tr, xr, vr = method(n, dt, x0, v0, w ** 2, sig / m)
        x_sum += xr
        x_square += xr ** 2

        v_sum += vr
        v_square += vr ** 2

        xv += xr * vr

    return tr, x_sum, x_square, v_sum, v_square, xv


def averageRunsData(n_runs, variables, add=""):
    if add != "":
        add = "_" + str(add)
    tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, rk4Stocastic)
    np.savez_compressed(f"Dat/{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                        xv=xv)
    np.savez(f"Dat/{n_runs}_variables{add}", variables)


def average(dat, N):
    return dat / N


def var(dat, dats, N):
    return (-(dat ** 2 / N) + dats) / N


def energy(xs, vs, N, m, w):
    return 0.5 * m * (vs / N) + 0.5 * m * w ** 2 * (xs / N)


def energyExpect(variables):
    n, dt, x0, v0, w, sig, m = variables
    tl = np.asarray([i * dt for i in range(int(n))])
    return 0.5 * m * v0 ** 2 + 0.5 * m * w ** 2 * x0 ** 2 + sig ** 2 * tl / (2 * m)


def energyVarExpect(variables):
    n, dt, x0, v0, w, sig, m = variables
    tl = np.asarray([i * dt for i in range(int(n))])
    return (sig**4/(4*m**2)) * (tl**2+(1-np.cos(2*w*tl))/(2*w**2))


def loadData(n_runs, add=""):
    if add != "":
        add = "_" + add

    dat = np.load(f"Dat/{n_runs}_variables{add}.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    dat = np.load(f"Dat/{n_runs}{add}.npz")
    x_sum = dat["x_sum"]
    x_square = dat["x_square"]
    v_sum = dat["v_sum"]
    v_square = dat["v_square"]
    xv = dat["xv"]



    tl = [i * dt for i in range(int(n))]

    x_average = average(x_sum, n_runs)
    v_average = average(v_sum, n_runs)
    x_variance = var(x_sum, x_square, n_runs)
    v_variance = var(v_sum, v_square, n_runs)
    sim_energy = energy(x_square, v_square, n_runs, m, w)
    return tl, x_average, v_average, x_variance, v_variance, sim_energy, var_list


def loadDataEnergy(n_runs, add=""):
    if add != "":
        add = "_" + add

    dat = np.load(f"Dat/{n_runs}_variables{add}.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    E_expect = energyExpect(var_list)
    E_var_epect = energyVarExpect(var_list)


    dat = np.load(f"Dat/{n_runs}{add}.npz")
    x_sum = dat["x_sum"]
    x_square = dat["x_square"]
    v_sum = dat["v_sum"]
    v_square = dat["v_square"]
    xv = dat["xv"]

    tl = [i * dt for i in range(int(n))]

    x_four = x_square ** 2
    v_four = v_square ** 2
    E_average = 0.5 * m * (v_square / n_runs + w ** 2 * x_square / n_runs)
    E_square = 0.25*m**2 * (v_four/n_runs)+0.25*m**2*w**4 * (x_four/n_runs) + 0.5*m**2*w**2*(v_square*x_square)/n_runs
    E_var = E_square-E_average**2
    return tl, E_var, E_average,E_var_epect, E_expect


def loadDataList(n_list, add=""):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Difference n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, add=add)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, x_average - xHarmonic(x0, v0, w, tl), label=f"{n_runs}")
        ax2.plot(tl, v_average - vHarmonic(x0, v0, w, tl))

    ax1.legend(loc=3)
    ax1.set_ylabel("x-x_expected")
    ax2.set_ylabel("v-v_expected")
    ax2.set_xlabel("t")
    plt.savefig("Spring_Diff")
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig.suptitle("x variance n")
    # for n_runs in n_list:
    #     tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, add=add)
    #     n, dt, x0, v0, w, sig, m = variables
    #     ax1.plot(tl, x_variance, label=f"{n_runs}")
    #     ax2.plot(tl, x_variance - xVarHarmonic(sig, m, w, tl))
    #
    # ax1.plot(tl, xVarHarmonic(sig, m, w, tl), label="Expected", color="k", linestyle="--")
    # ax1.legend(loc=3)
    # ax1.set_ylabel("x var")
    # ax2.set_ylabel("x var-expected")
    # ax2.set_xlabel("t")
    # plt.savefig("Spring_var")
    # plt.show()
    #
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig.suptitle("v variance n")
    # for n_runs in n_list:
    #     tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, add=add)
    #     n, dt, x0, v0, w, sig, m = variables
    #     ax1.plot(tl, v_variance, label=f"{n_runs}")
    #     ax2.plot(tl, v_variance - vVarHarmonic(sig, m, w, tl))
    #
    # ax1.plot(tl, vVarHarmonic(sig, m, w, tl), label="Expected", color="k", linestyle="--")
    # ax1.legend(loc=3)
    # ax1.set_ylabel("v var")
    # ax2.set_ylabel("v var-expected")
    # ax2.set_xlabel("t")
    # plt.savefig("Spring_var_v")
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Energy n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, add=add)
        ax1.plot(tl, sim_energy, label=f"{n_runs}")
        ax2.plot(tl, sim_energy - energyExpect(variables))

    ax1.plot(tl, energyExpect(variables), label="Expected", color="k", linestyle="--")
    ax2.plot(tl, [0 for i in tl], color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Difference")
    ax2.set_xlabel("t")
    plt.savefig("Spring_energ")
    plt.show()


def loadDataListEnergy(n_list, add=""):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for n_run in n_runs_list:
        tl, ev, ea, ev_expec, ea_expec = loadDataEnergy(n_run, add=add)
        ax1.plot(tl, ea, label=f"{n_run}")

        ax2.plot(tl, ev)
        # ax2.plot(tl,ev_expec)
        # ax2.plot(tl, ev_expec*5*10**9, linestyle=":")

    ax1.plot(tl, ea_expec, linestyle=":")
    ax1.legend()

    plt.show()


def phase(variables, T):
    n, dt, x0, v0, w, sig, m = variables
    return T * (m * w ** 2 * x0 ** 2 / (2 * hbar) + m * v0 ** 2 / (2 * hbar) + sig ** 2 * T / (4 * m * hbar))


def phase_var(variables, T):
    n, dt, x0, v0, w, sig, m = variables
    print(sig ** 2 / hbar ** 2)
    print(v0 ** 2 + w ** 2 * x0 ** 2)
    print(T ** 3 / 6 - T * np.cos(2 * w * T) / (4 * w ** 2) + np.sin(2 * w * T) / (8 * w ** 3))
    return (sig ** 2 / hbar ** 2) * (v0 ** 2 + w ** 2 * x0 ** 2) * (
            T ** 3 / 6 - T * np.cos(2 * w * T) / (4 * w ** 2) + np.sin(2 * w * T) / (8 * w ** 3))


def phasePlot(variables):
    n, dt, x0, v0, w, sig, m = variables
    Tl = np.linspace(0.1, 1, 100)
    # plt.figure()
    # plt.title("Additional phase for differing travel time")
    # plt.plot(Tl, phase(variables, Tl))
    # plt.ylabel("Phase (s^-1)")
    # plt.xlabel("Travel time (s)")
    # plt.show()

    plt.figure()
    plt.title("Variance of phase")
    plt.plot(Tl, phase_var(variables, Tl))
    plt.ylabel("Varof Phase/Phase")
    plt.xlabel("Travel time (s)")
    plt.show()


def phaseVariance(variables, T):
    n, dt, x0, v0, w, sig, m = variables
    coef1 = sig ** 4 / (4 * m ** 2 * hbar ** 2)
    coef2 = (sig ** 2 / hbar ** 2) * (v0 ** 2 + w ** 2 * x0 ** 2)
    phasevar1 = coef1 * (T ** 4 / 12 + T ** 2 / (2 * w ** 2) + (np.cos(2 * w * T) - 1) / (4 * w ** 4))
    phasevar2 = coef2 * (T ** 3 / 6 - (T * np.cos(2 * w * T)) / (8 * w ** 2) + (np.sin(2 * w * T) / (16 * w ** 3)))
    return phasevar1 + phasevar2


def phase(variables, T):
    n, dt, x0, v0, w, sig, m = variables
    return (w ** 2 * x0 ** 2 + v0 ** 2) * (T / 2 * hbar) + sig ** 2 * T ** 2 / (4 * m * hbar)


def phasePlots(variables, sigList, T):
    n, dt, x0, v0, w, sig, m = variables
    print(x0, v0, w, m)
    textstr = rf'$x_0$={x0}'+'\n'+rf'$v_0$={v0}'+'\n'+rf'$\omega$={w}'+'\n'+rf'$m$={m:.2E}'
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Phase vs Time for varying Sigma")
    for s in sigList:
        variables = [n, dt, x0, v0, w, s, m]
        phase1 = phase(variables, T)
        phase2 = phaseVariance(variables, T)
        ax1.plot(T, phase1, label=f"{s:.2E}")
        ax2.plot(T, phase2)

    ax1.legend(framealpha=1)
    ax1.set_ylabel("Phase")
    ax2.set_ylabel("Phase Variance")
    ax2.set_xlabel("Time")

    anchored = AnchoredText(textstr,loc=2)
    anchored.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(anchored)
    plt.savefig("Phase_Variance_Initial")
    plt.show()


#################################
d = 0.004
v = 0.01
t = d / v
print("Final time", t)
n = 1000
dt = t / n
print("dt", dt)

m = 1.44 * 10 ** (-25)
w = 6.1 * 10 ** 2
k = w ** 2 * m

print("t0", w ** (-1))
hbar = 1.0545718 * 10 ** -34

sig = m / 100

x0 = 0
v0 = 0

var_list = [n, dt, x0, v0, w, sig, m]

# detectFolder()
# phasePlot(var_list)

loadData(10000)

############## N runs Data
# n_runs_list = [1000, 10000, 50000, 100000]
# n_runs_list = [100000]
# for n_runs in n_runs_list:
#     averageRunsData(n_runs, var_list, add="in")
# loadDataList(n_runs_list,add="")
# loadDataList(n_runs_list, add=f"{dt}")


######## Energy Sim Var
# n_runs_list = [1000, 10000, 50000, 100000]
# n_runs_list = [10000]
# loadDataListEnergy(n_runs_list, add="dt")



################ Time Steps
# n_runs_list = [10000]
# loadDataListEnergy(n_runs_list,add=f"0.0008")
# loadDataListEnergy(n_runs_list,add=f"0.0004")
# loadDataListEnergy(n_runs_list,add="4e-05")


#####Phase Variance
# x0=0.0001
# v0=0.001
# sig_list = [2 * sig, sig, sig / 2]
# t = np.asarray([i * dt for i in range(n)])
#
# phasePlots(var_list, sig_list, t)

# phasevar = phaseVariance(var_list, t)
# plt.title("Variance of Atomic Phase")
# plt.plot(t, phase(var_list, t))
# plt.ylabel("Atomic Phase")
# plt.xlabel("Time")
# plt.show()
#
# plt.title("Variance of Atomic Phase")
# plt.plot(t, phasevar)
# plt.ylabel("Atomic Phase Variance")
# plt.xlabel("Time")
# plt.show()
