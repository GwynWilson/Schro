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


def rk4StocasticPhase(variables, noise=None):
    n, dt, x0, v0, w, sig, m = variables
    a = w ** 2
    b = sig / m
    xl = [x0]
    vl = [v0]
    pl = [0]
    varl = [0]
    varl2 = [0]
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


def averageRuns(n_runs, variables, method, norm=True):
    n, dt, x0, v0, w, sig, m = variables
    if norm:
        x_sum = np.zeros(n)
        x_square = np.zeros(n)
        v_sum = np.zeros(n)
        v_square = np.zeros(n)
        xv = np.zeros(n)
        xv_square = np.zeros(n)
        x_four = np.zeros(n)
        v_four = np.zeros(n)
        earr = np.zeros((n, n))

    else:
        energy_sum = np.zeros(n)
        earr = np.zeros((n, n))
        phase = np.zeros(n)
        phase_sq = np.zeros(n)

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

            if norm:
                tr, xr, vr = method(n, dt, x0, v0, w ** 2, sig / m)

            else:
                tr, xr, vr, pr, psr = method(variables)
        energy = np.array([0.5 * m * vr ** 2 + 0.5 * m * w ** 2 * xr ** 2])
        energtT = np.transpose(energy)
        energyarr = np.dot(energtT, energy)
        earr += energyarr

        if norm:
            x_sum += xr
            x_square += xr ** 2
            x_four += xr ** 4

            v_sum += vr
            v_square += vr ** 2
            v_four += vr ** 4

            xv += xr * vr
            xv_square += xr ** 2 * vr ** 2
        else:
            energy_sum += energy[0]
            phase += pr
            phase_sq += psr

    if norm:
        return tr, x_sum, x_square, v_sum, v_square, xv, xv_square, x_four, v_four, earr
    else:
        return tr, energy_sum, phase, phase_sq, earr


def averageRunsData(n_runs, variables, add=""):
    if add != "":
        add = "_" + str(add)
    tr, x_sum, x_square, v_sum, v_square, xv, xv_square, x_four, v_four, earr = averageRuns(n_runs, variables,
                                                                                            rk4Stocastic)
    np.savez_compressed(f"Dat/{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                        xv=xv, xv_square=xv_square, x_four=x_four, v_four=v_four)
    np.savez(f"Dat/{n_runs}_variables{add}", variables)
    np.savez(f"Dat/{n_runs}_earr{add}", earr)


def averageRunsDataPhase(n_runs, variables, add="_phase"):
    if add != "_phase":
        add = "_phase_" + str(add)

    tr, energy_sum, phase_av, phase_sq, earr = averageRuns(n_runs, variables, rk4StocasticPhase, norm=False)
    np.savez_compressed(f"Dat/{n_runs}{add}", tl=tr, energy=energy_sum, phase=phase_av, phase_sq=phase_sq)
    np.savez(f"Dat/{n_runs}_variables{add}", variables)
    np.savez(f"Dat/{n_runs}_earr{add}", earr)


def energyExpect(variables):
    n, dt, x0, v0, w, sig, m = variables
    tl = np.asarray([i * dt for i in range(int(n))])
    return 0.5 * m * v0 ** 2 + 0.5 * m * w ** 2 * x0 ** 2 + sig ** 2 * tl / (2 * m)


def energyVarExpect(variables):
    n, dt, x0, v0, w, sig, m = variables
    tl = np.asarray([i * dt for i in range(int(n))])
    return (sig ** 4 / (4 * m ** 2)) * (tl ** 2 + (1 - np.cos(2 * w * tl)) / (2 * w ** 2))


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
    xv_square = dat["xv_square"]
    x_four = dat["x_four"]
    v_four = dat["v_four"]

    dat2 = np.load(f"Dat/{n_runs}_earr{add}.npz")
    earr = dat2["arr_0"]
    diagonal = np.diagonal(earr) / n_runs

    tl = [i * dt for i in range(int(n))]

    E_average = 0.5 * m * (v_square / n_runs + w ** 2 * x_square / n_runs)
    E_square = 0.25 * m ** 2 * (v_four / n_runs) + 0.25 * m ** 2 * w ** 4 * (
            x_four / n_runs) + 0.5 * m ** 2 * w ** 2 * (xv_square) / n_runs
    E_var = E_square - E_average ** 2

    return tl, E_var, E_average, E_var_epect, E_expect


def loadDataPhase(n_runs, add="_phase", inc=False):
    if add != "_phase":
        add = "_phase_" + str(add)

    dat = np.load(f"Dat/{n_runs}_variables{add}.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    dat = np.load(f"Dat/{n_runs}{add}.npz")
    tl = dat["tl"]
    energy = dat["energy"]
    phase = dat["phase"]
    phase_sq = dat["phase_sq"]

    if inc != False:
        arr_phase, arr_var, arr_tl = loadArray(n_runs, inc, add="phase", var=True)

    phase_av = phase / n_runs
    phase_sq = phase_sq / n_runs
    phase_var = (phase_sq - phase_av ** 2) / (hbar ** 2)
    phase_av = phase_av / hbar

    if inc != False:
        return tl, phase_av, phase_var, arr_tl, arr_phase, arr_var
    else:
        return tl, phase_av, phase_var


def phaseIntegration(earr, tl, num):
    temp = np.zeros(int(num))
    for ind in range(num):
        temp[ind] = simps(earr[ind][:num], tl[:num])
    phase_diff = simps(temp, tl[:num])
    return phase_diff


def loadArray(n_runs, inc=1, add="", sample=None, var=False):
    if add == "phase":
        pload = True
    if add != "":
        add = "_" + add

    dat = np.load(f"Dat/{n_runs}_variables{add}.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    if sample != None:
        inc = int(n / sample)

    dat2 = np.load(f"Dat/{n_runs}_earr{add}.npz")
    earr = dat2["arr_0"] / n_runs

    tl = [i * dt for i in range(int(n))]

    if pload:
        dat = np.load(f"Dat/{n_runs}{add}.npz")
        E_average = dat["energy"] / n_runs
    else:
        dat = np.load(f"Dat/{n_runs}{add}.npz")
        x_square = dat["x_square"]
        v_square = dat["v_square"]
        E_average = 0.5 * m * (v_square / n_runs + w ** 2 * x_square / n_runs)

    phase_av = np.zeros(int(n))
    phase_sq = np.zeros(int(n))
    for i in range(1, int(n), inc):
        if i % 1000 == 1:
            print(i)
        phase_av[i] = simps(E_average[:i], tl[:i]) / hbar
        if var:
            phase_sq[i] = phaseIntegration(earr, tl, i) / (hbar ** 2)

    phase_var = phase_sq - phase_av ** 2
    return phase_av[1::inc], phase_var[1::inc], tl[1::inc]


def averagePhaseTheory(variables, tl):
    n, dt, x0, v0, w, sig, m = variables
    T = np.asarray(tl)
    return (m * w ** 2 * x0 ** 2 / (2 * hbar) + m * v0 ** 2 / (2 * hbar)) * T + sig ** 2 * T ** 2 / (4 * m * hbar)


def variancePhaseTheory(variables, tl):
    n, dt, x0, v0, w, sig, m = variables
    T = np.asarray(tl)
    coeff1 = sig ** 2 * (v0 ** 2 + w ** 2 * x0 ** 2) / (hbar ** 2)
    coeff2 = (sig ** 2 / (2 * m * hbar)) ** 2
    return coeff1 * (T ** 3 / 6 - T * np.cos(2 * w * T) / (8 * w ** 2) + np.sin(2 * w * T) / (16 * w ** 3)) \
           + coeff2 * (T ** 4 / 12 + T ** 2 / (2 * w ** 2) + (np.cos(2 * w * T) - 1) / (4 * w ** 2))


def loadArrayTest(n_runs, inc=1, add=""):
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
    xv_square = dat["xv_square"]
    x_four = dat["x_four"]
    v_four = dat["v_four"]

    dat2 = np.load(f"Dat/{n_runs}_earr{add}.npz")
    earr = dat2["arr_0"]
    diagonal = np.diagonal(earr) / n_runs

    E_square = 0.25 * m ** 2 * (v_four / n_runs) + 0.25 * m ** 2 * w ** 4 * (
            x_four / n_runs) + 0.5 * m ** 2 * w ** 2 * (xv_square) / n_runs

    tl = [i * dt for i in range(int(n))]
    plt.title(f"Energy Squared n={n_runs} dt={dt}")
    plt.plot(tl, diagonal, label="Array")
    plt.plot(tl, E_square, linestyle=":", label="Simulation")
    plt.plot(tl, energyVarExpect(var_list) + energyExpect(var_list) ** 2, label="theory")
    plt.legend()
    plt.savefig(f"Energy_Squared_{dt}")
    plt.show()
    return 0


def loadList(input, sample, variables):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    for i in input:
        n_runs, add = i
        print(f"Number of Runs {n_runs}")
        phase_av, phase_var, tl = loadArray(n_runs, sample=sample, add=add)
        ax1.plot(tl, phase_av, label=f"{n_runs}")
        ax2.plot(tl, abs(phase_av - averagePhaseTheory(variables, tl)))

    ax1.legend()
    # plt.savefig("Average_Phase_Comparison")
    plt.show()
    return 0


def loadListPhase(input, variables):
    dat = np.load(f"Dat/{input[0]}_variables_phase.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    variables = [n, dt, x0, v0, w, sig, m]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    fig.suptitle(f"Average Phase dt={dt}")
    for i in input:
        n_runs = i
        print(f"Number of Runs {n_runs}")
        # phase_av, phase_var, tl = loadArray(n_runs, sample=sample, add=add)
        tl, av, var = loadDataPhase(i, add="_phase")
        ax1.plot(tl, av, label=f"{n_runs}")
        ax2.plot(tl, av - averagePhaseTheory(variables, tl))

    ax1.plot(tl, averagePhaseTheory(variables, tl), linestyle=":", color="k", label="Theory")
    ax1.legend()
    ax1.set_ylabel("Average Phase")
    ax2.set_ylabel("Average Phase Difference")
    ax2.set_xlabel("Time")
    plt.savefig(f"Average_Phase_dt={dt}.png")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(f"Phase Variance dt={dt}")
    for i in input:
        n_runs = i
        print(f"Number of Runs {n_runs}")
        # phase_av, phase_var, tl = loadArray(n_runs, sample=sample, add=add)
        tl, av, var = loadDataPhase(i, add="_phase")
        ax1.plot(tl, var, label=f"{n_runs}")
        ax2.plot(tl, var - variancePhaseTheory(variables, tl))

    ax1.plot(tl, variancePhaseTheory(variables, tl), linestyle=":", color="k", label="Theory")
    ax1.legend()
    ax1.set_ylabel("Phase Variance")
    ax2.set_ylabel("Phase Variance Difference")
    ax2.set_xlabel("Time")
    plt.savefig(f"Variance_Phase_dt={dt}.png")
    plt.show()
    return 0


d = 0.004
v = 0.01
t = d / v
print("Final time", t)
n = 500
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

n_runs = 1000

########## Energy variance
# data_list = [2000, 4000, 6000, 8000]
# for i in data_list:
#     averageRunsData(i, var_list)
# tl, E_var, E_average, E_var_epect, E_expect = loadDataEnergy(n_runs)
# simps(E_var,dt)
#
# plt.plot(tl, E_average)
# plt.plot(tl, E_expect, linestyle=":")
# plt.show()
#
# plt.plot(tl, E_var)
# plt.plot(tl, E_var_epect, linestyle=":")
# plt.show()


######## Energy array
# averageRunsData(n_runs, var_list, add=f"Runs_{n_runs}")
# sample = 200
# inc = int(n / sample)

# loadArrayTest(n_runs, add=f"Runs_{n_runs}")

# phase_av, phase_var, tl = loadArray(n_runs, inc, add=f"Runs_{n_runs}")
#
# plt.figure()
# plt.title(f"Average Phase Difference dt={dt}")
# # plt.plot(tl, phase_av, label="Simulated")
# # plt.plot(tl, averagePhaseTheory(var_list, tl),linestyle=":", label="Theory")
#
# plt.plot(tl, abs(phase_av-averagePhaseTheory(var_list, tl)))
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.legend()
# plt.savefig("Phase_Average_Difference")
# plt.show()
#
# plt.title(f"Phase Variance Difference dt={dt}")
# # plt.plot(tl, phase_var, label="Simulated")
# # plt.plot(tl, variancePhaseTheory(var_list, tl), label="Theory")
#
# plt.plot(tl, abs(phase_var-variancePhaseTheory(var_list, tl)))
# plt.xlabel("Time")
# plt.ylabel("Phase Variance")
# plt.legend()
# plt.savefig("Phase_Variance_Difference")
# plt.show()


############ List
# inp = [(2000, ""), (4000, ""), (6000, ""), (8000, ""), (10000, "Runs_10000")]
#
# sample = 1000
#
# loadList


########### Phase Testing
# tl, xl, vl, pl = rk4StocasticPhase(var_list)
# El = 0.5 * m * w ** 2 * xl ** 2 + 0.5 * m * vl ** 2
# temp = [0]
# for i in range(1, len(tl)):
#     temp.append(simps(El[:i], tl[:i])/hbar)
#
# plt.plot(tl, pl)
# plt.plot(tl, temp)
# plt.show()


# tl, xl, vl, pl, varl = rk4StocasticPhase(var_list)
# phase_var = (varl - pl ** 2) / (hbar ** 2)
#
# energy = np.array([0.5 * m * vl ** 2 + 0.5 * m * w ** 2 * xl ** 2])
# energtT = np.transpose(energy)
# energyarr = np.dot(energtT, energy)
#
# sample = 10
# inc = int(n / sample)
#
# phase_av = np.zeros(int(n))
# phase_sq = np.zeros(int(n))
# for i in range(1, int(n), inc):
#     if i % 1000 == 1:
#         print(i)
#     phase_av[i] = simps(energy[0][:i], tl[:i]) / hbar
#     phase_sq[i] = phaseIntegration(energyarr, tl, i) / (hbar ** 2)
#
# phase_var2 = (phase_sq - phase_av ** 2) / hbar ** 2
#
# plt.plot(tl, phase_var)
# plt.plot(tl, phase_var2)
# plt.show()


################## Phase Data

# tl, av, var = loadDataPhase(1000, add="_phase")
# plt.plot(tl, av)
# plt.plot(tl, averagePhaseTheory(var_list, tl))
# plt.show()

# sample = 1000
# inc = int(n / sample)

# tl, av, var, arr_tl, arr_av, arr_var = loadDataPhase(10000, add="_phase", inc=inc)
# plt.title("Average Phase")
# plt.plot(tl, av, label="Differentail")
# plt.plot(arr_tl, arr_av, label="Simpsons")
# plt.plot(tl, averagePhaseTheory(var_list, tl), label="Theoretical")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.savefig("Method Comparison")
# plt.show()
#
# plt.title("Average Phase difference")
# plt.plot(tl, av-averagePhaseTheory(var_list, tl), label="Differentail")
# plt.plot(arr_tl, arr_av-averagePhaseTheory(var_list, tl)[1::inc], label="Simpsons")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.savefig("Method Comparison Difference")
# plt.show()


# tl, av, var, arr_tl, arr_av, arr_var = loadDataPhase(10000, add="_phase", inc=inc)
# plt.title("Phase Variance")
# plt.plot(tl, var, label="Differentail")
# plt.plot(arr_tl, arr_var, label="Simpsons")
# plt.plot(tl, variancePhaseTheory(var_list, tl), label="Theoretical")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.savefig("Method Comparison Variance")
# plt.show()
#
# plt.title("Phase Variance difference")
# plt.plot(tl, var-variancePhaseTheory(var_list, tl), label="Differentail")
# plt.plot(arr_tl, arr_var-variancePhaseTheory(var_list, tl)[1::inc], label="Simpsons")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.savefig("Method Comparison Difference Variance")
# plt.show()


###### Phase Data List
# phase_run_list = [20000, 40000, 60000, 80000]
# for i in phase_run_list:
#     averageRunsDataPhase(i, var_list)
# loadListPhase(phase_run_list, var_list)

# sample = 1000
# inc = int(n / sample)
# tl, av, var, arr_tl, arr_av, arr_var = loadDataPhase(4000, add="_phase", inc=inc)
# plt.title("Average Phase")
# plt.plot(tl, av, label="Differentail")
# plt.plot(arr_tl, arr_av, label="Simpsons")
# plt.plot(tl, averagePhaseTheory(var_list, tl), label="Theoretical")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Phase")
# plt.savefig("Method Comparison")
# plt.show()
