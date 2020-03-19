import numpy as np
import matplotlib.pyplot as plt
import time


def genNoise(n, dt):
    """
    Will generate array of length 2n of gaussian white noise
    :param n:
    :return:
    """
    return np.sqrt(dt) * np.random.randn(2 * n)


def rk2Stocastic(n, dt, x0, v0, a, b, noise=None):
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
        t += dt
        tl.append(t)

        x += vol * dt - (a * xol * dt ** 2) / 2 + (b * noise[2 * i] * dt) / 2
        # x += vol * dt - (a * xol * dt ** 2)
        xl.append(x)

        v += -a * xol * dt + b * noise[2 * i + 1] - (a * vol * dt ** 2) / 2
        # v += -a * xol * dt + b * noise[2 * w + 1]
        vl.append(v)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def eulerStocastic(n, dt, x0, v0, a, b, noise=None):
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
    for w in range(n - 1):
        t += dt
        tl.append(t)

        x += vol * dt
        xl.append(x)

        v += -a * xol * dt + b * noise[2 * w + 1]
        # v += -a * x * dt
        vl.append(v)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def oneRunNotStocastic(n, dt, x0, v0, a, b):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = v0
    for w in range(n - 1):
        t += dt
        tl.append(t)

        x += vol * dt - (a * xol * dt ** 2) / 2
        xl.append(x)

        v += -a * xol * dt - (a * vol * dt ** 2) / 2
        # v += -a * x * dt
        vl.append(v)

        xol = x
        vol = v

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


def compareEulerRk2(variables):
    n, dt, x0, v0, w, sig, m = variables
    noise = genNoise(n, dt)

    tr, xr, vr = rk2Stocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
    te, xe, ve = eulerStocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)

    plt.plot(tr, xr, label="RK")
    plt.plot(te, xe, label="Euler")
    plt.plot(tr, xHarmonic(x0, v0, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.show()

    plt.plot(tr, xr - xHarmonic(x0, v0, w, tr), label="RK")
    plt.plot(tr, xe - xHarmonic(x0, v0, w, tr), label="Euler")
    plt.legend(loc=3)
    plt.show()


def runsData(data):
    print(np.shape(data))
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        if i % 10000 == 0:
            print(i)
        dslice = data[:, i]
        average[i] = np.mean(dslice)
        variance[i] = np.var(dslice)
    return average, variance


def compareEulerRk2ManyRun(n_runs, variables):
    n, dt, x0, v0, w, sig, m = variables
    r_x_average = np.zeros(n)
    r_x_square = np.zeros(n)
    r_v_average = np.zeros(n)
    r_v_square = np.zeros(n)

    e_x_average = np.zeros(n)
    e_x_square = np.zeros(n)
    e_v_average = np.zeros(n)
    e_v_square = np.zeros(n)
    for n_run in range(n_runs):
        if n_run % 100 == 0:
            print(f"Run: {n_run}")
        noise = genNoise(n, dt)
        tr, xr, vr = rk2Stocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
        te, xe, ve = eulerStocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)

        r_x_average += xr
        r_x_square += xr ** 2

        r_v_average += vr
        r_v_square += vr ** 2

        e_x_average += xe
        e_x_square += xe ** 2

        e_v_average += ve
        e_v_square += ve ** 2

    r_x_mean = r_x_average / n_runs
    r_x_variance = (-(r_x_average ** 2 / n_runs) + r_x_square) / n_runs
    r_v_mean = r_v_average / n_runs
    r_v_variance = (-(r_v_average ** 2 / n_runs) + r_v_square) / n_runs

    e_x_mean = e_x_average / n_runs
    e_x_variance = (-(e_x_average ** 2 / n_runs) + e_x_square) / n_runs
    e_v_mean = e_v_average / n_runs
    e_v_variance = (-(e_v_average ** 2 / n_runs) + e_v_square) / n_runs

    plt.title(f"Comparsion x Position n={n}, runs={n_runs}")
    plt.plot(tr, r_x_mean, label="RK")
    plt.plot(te, e_x_mean, label="Euler")
    plt.plot(tr, xHarmonic(x0, v0, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xpos_{n}_{n_runs}")
    plt.show()

    plt.title(f"Comparsion x Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_mean - xHarmonic(x0, v0, w, tr), label="RK")
    plt.plot(tr, e_x_mean - xHarmonic(x0, v0, w, tr), label="Euler")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xpos_difference_{n}_{n_runs}")
    plt.show()

    plt.title(f"Comparsion x Variance n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance, label="RK")
    plt.plot(te, e_x_variance, label="Euler")
    plt.plot(tr, xVarHarmonic(sig, m, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_{n}_{n_runs}")
    plt.show()

    plt.title(f"Comparsion x Variance Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance - xVarHarmonic(sig, m, w, tr), label="RK")
    plt.plot(tr, e_x_variance - xVarHarmonic(sig, m, w, tr), label="Euler")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_difference_{n}_{n_runs}")
    plt.show()


def compareEulerRk2ManyRun2(n_runs, variables):
    n, dt, x0, v0, w, sig, m = variables
    r_x_average_pre = np.zeros([n_runs, n])
    r_v_average_pre = np.zeros([n_runs, n])
    e_x_average_pre = np.zeros([n_runs, n])
    e_v_average_pre = np.zeros([n_runs, n])

    for n_run in range(n_runs):
        if n_run % 100 == 0:
            print(f"Run: {n_run}")
        noise = genNoise(n, dt)
        tr, xr, vr = rk2Stocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
        te, xe, ve = eulerStocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
        r_x_average_pre[n_run] = xr
        r_v_average_pre[n_run] = vr
        e_x_average_pre[n_run] = xe
        e_v_average_pre[n_run] = ve

    r_x_average, r_x_variance = runsData(r_x_average_pre)
    e_x_average, e_x_variance = runsData(e_x_average_pre)

    plt.title(f"Comparsion x Position n={n}, runs={n_runs}")
    plt.plot(tr, r_x_average, label="RK")
    plt.plot(te, e_x_average, label="Euler")
    plt.plot(tr, xHarmonic(x0, v0, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xpos_{n}_{n_runs}_alt")
    plt.show()

    plt.title(f"Comparsion x Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_average - xHarmonic(x0, v0, w, tr), label="RK")
    plt.plot(tr, e_x_average - xHarmonic(x0, v0, w, tr), label="Euler")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xpos_difference_{n}_{n_runs}_alt")
    plt.show()

    plt.title(f"Comparsion x Variance n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance, label="RK")
    plt.plot(te, e_x_variance, label="Euler")
    plt.plot(tr, xVarHarmonic(sig, m, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_{n}_{n_runs}_alt")
    plt.show()

    plt.title(f"Comparsion x Variance Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance - xVarHarmonic(sig, m, w, tr), label="RK")
    plt.plot(tr, e_x_variance - xVarHarmonic(sig, m, w, tr), label="Euler")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_difference_{n}_{n_runs}_alt")
    plt.show()


def manyRunVarianceComparison(n_runs, variables):
    n, dt, x0, v0, w, sig, m = variables
    r_x_average_pre = np.zeros([n_runs, n])

    r_x_average_sum = np.zeros(n)
    r_x_square_sum = np.zeros(n)

    for n_run in range(n_runs):
        if n_run % 100 == 0:
            print(f"Run: {n_run}")
        noise = genNoise(n, dt)
        tr, xr, vr = rk2Stocastic(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
        r_x_average_pre[n_run] = xr

        r_x_average_sum += xr
        r_x_square_sum += xr ** 2

    r_x_mean = r_x_average_sum / n_runs
    r_x_variance_sum = (-(r_x_average_sum ** 2 / n_runs) + r_x_square_sum) / n_runs

    r_x_average, r_x_variance = runsData(r_x_average_pre)
    plt.plot(tr, r_x_variance_sum, label="Sum")
    plt.plot(tr, r_x_variance, label="Data")
    plt.show()


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


def averageRunsData(n_runs, variables, add=""):
    if add != "":
        add = "_" + str(add)
    tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, rk2Stocastic)
    np.savez_compressed(f"Dat/rk_{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                        xv=xv)
    np.savez(f"Dat/rk_{n_runs}_variables{add}", variables)


def loadData(n_runs, plot=False):
    dat = np.load(f"Dat/rk_{n_runs}_variables.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    dat = np.load(f"Dat/rk_{n_runs}.npz")
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
    if plot:
        plotData(x_average, v_average, x_variance, sim_energy, var_list)
    return tl, x_average, v_average, x_variance, v_variance, sim_energy, var_list


def loadDataList(n_list):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, x_average - xHarmonic(x0, v0, w, tl), label=f"{n_runs}")
        ax2.plot(tl, v_average - vHarmonic(x0, v0, w, tl))

    ax1.legend(loc=3)
    ax1.set_ylabel("x-x_expected")
    ax2.set_ylabel("v-v_expected")
    ax2.set_xlabel("t")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, x_variance, label=f"{n_runs}")
        ax2.plot(tl, x_variance - xVarHarmonic(sig, m, w, tl))

    ax1.plot(tl, xVarHarmonic(sig, m, w, tl))
    ax1.legend(loc=3)
    ax1.set_ylabel("x var")
    ax2.set_ylabel("x var-expected")
    ax2.set_xlabel("t")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs)
        ax1.plot(tl, sim_energy, label=f"{n_runs}")
        ax2.plot(tl, sim_energy - energyExpect(variables))

    ax1.plot(tl, energyExpect(variables))
    ax1.legend(loc=3)
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Difference")
    ax2.set_xlabel("t")
    plt.show()


def multipleData(rep, n_runs, variables):
    for i in range(rep):
        print(f"----------------------------------------------\nRepetition {i}")
        averageRunsData(n_runs, variables, add=i)

def mergeData(rep, n_runs,n):
    x_sum = np.zeros(n)
    x_square = np.zeros(n)
    v_sum = np.zeros(n)
    v_square = np.zeros(n)
    xv = np.zeros(n)
    for i in range(rep):
        dat = np.load(f"Dat/rk_{n_runs}_{i}.npz")
        x_sum += dat["x_sum"]
        x_square += dat["x_square"]
        v_sum += dat["v_sum"]
        v_square = +dat["v_square"]
        xv = +dat["xv"]

    dat = np.load(f"Dat/rk_{n_runs}_variables_0.npz")
    variables = dat["arr_0"]

    np.savez_compressed(f"Dat/rk_{n_runs*rep}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                        xv=xv)
    np.savez(f"Dat/rk_{n_runs*rep}_variables", variables)



def plotData(x_average, v_average, x_variance, sim_energy, variables):
    plt.plot(tl, x_average)
    plt.show()
    plt.plot(tl, x_variance)
    plt.show()
    plt.plot(tl, sim_energy)
    plt.plot(tl, energyExpect(var_list))
    plt.show()


def plotAverageRuns(n_runs, variables, method):
    tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, method)
    x_average = average(x_sum, n_runs)
    v_average = average(v_sum, n_runs)
    x_variance = var(x_sum, x_square, n_runs)
    v_variance = var(v_sum, v_square, n_runs)
    sim_energy = energy(x_square, v_square, n_runs, m, w)

    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.plot(tr, x_average)
    # ax2.plot(tr, v_average)
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.plot(tr, x_variance)
    # ax1.plot(tr, xVarHarmonic(sig, m, w, tr))
    # ax2.plot(tr, v_variance)
    # # plt.savefig(method.__name__ + "_Variance_BiggestRuns")
    # plt.show()
    #
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.plot(tl, sim_energy, label=f"{n_runs}")
    # ax2.plot(tl, sim_energy - energyExpect(variables))
    # plt.show()
    plt.plot(tl, 0.5 * m * v_square / n_runs)
    plt.plot(tl, 0.5 * m * w ** 2 * x_square / n_runs)
    plt.show()
    plt.plot(tl, 0.5 * m * v_square / n_runs + 0.5 * m * w ** 2 * x_square / n_runs)
    plt.show()


d = 5
v = 1
t = d / v
print("Final time", t)
n = 50000
dt = t / n
print("dt", dt)
tl = [i * dt for i in range(n)]

m = 1
w = 5
k = w ** 2 * m

print("t0", w ** (-1))

sig = 2

# x0 = 0.0002
x0 = 10
v0 = 0

n_runs = 100
var_list = [n, dt, x0, v0, w, sig, m]
# averageRuns(n_runs, var_list, rk2Stocastic)
# plotAverageRuns(n_runs, var_list, rk2Stocastic)

# compareEulerRk2(var_list)
# compareEulerRk2ManyRun(n_runs, var_list)

# manyRunVarianceComparison(n_runs, var_list)

# n_runs = 10000
# averageRunsData(n_runs, var_list)
# loadData(n_runs, plot=True)

# n_runs_list = [500, 1000, 5000, 10000]
# loadDataList(n_runs_list)


n_runs = 10000
rep = 5
# multipleData(rep, n_runs, var_list)
mergeData(rep,n_runs,n)