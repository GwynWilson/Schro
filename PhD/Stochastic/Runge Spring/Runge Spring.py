import numpy as np
import matplotlib.pyplot as plt
import time


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


def rk2Stocastic2(n, dt, x0, v0, a, b, noise=None):
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

        v += -a * xol * dt + b * noise[2 * i] - (a * vol * dt ** 2) / 2
        # v += -a * xol * dt + b * noise[2 * w + 1]
        vl.append(v)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


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


def rk4Stocastic2(n, dt, x0, v0, a, b, noise=None):
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
        l3 = dt * g(t + dt, x + k2, v + l2, a) + b * noise[2 * i]

        x += (k0 + 2 * k1 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6
        t += dt

        xl.append(x)
        vl.append(v)
        tl.append(t)

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


def vVarHarmonic(sig, m, w, t):
    t = np.asarray(t)
    return (sig ** 2 / (2 * (m ** 2))) * (t + np.sin(2 * w * t) / (2 * w))


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


def compareMethods(n_runs, variables, method1, method2, add=""):
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
        tr, xr, vr = method1(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
        te, xe, ve = method2(n, dt, x0, v0, w ** 2, sig / m, noise=noise)

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

    plt.figure()
    plt.title(f"Comparsion x Position n={n}, runs={n_runs}")
    plt.plot(tr, r_x_mean, label=method1.__name__)
    plt.plot(te, e_x_mean, label=method2.__name__)
    plt.plot(tr, xHarmonic(x0, v0, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"comparison_xpos_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion x Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_mean - xHarmonic(x0, v0, w, tr), label=method1.__name__)
    plt.plot(tr, e_x_mean - xHarmonic(x0, v0, w, tr), label=method2.__name__)
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xpos_difference_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion x Variance n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance, label=method1.__name__)
    plt.plot(te, e_x_variance, label=method2.__name__)
    plt.plot(tr, xVarHarmonic(sig, m, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion x Variance Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_x_variance - xVarHarmonic(sig, m, w, tr), label=method1.__name__)
    plt.plot(tr, e_x_variance - xVarHarmonic(sig, m, w, tr), label=method2.__name__)
    plt.legend(loc=3)
    plt.savefig(f"Comparison_xvar_difference_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion v Position n={n}, runs={n_runs}")
    plt.plot(tr, r_v_mean, label=method1.__name__)
    plt.plot(te, e_v_mean, label=method2.__name__)
    plt.plot(tr, vHarmonic(x0, v0, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"comparison_vpos_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion v Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_v_mean - vHarmonic(x0, v0, w, tr), label=method1.__name__)
    plt.plot(tr, e_v_mean - vHarmonic(x0, v0, w, tr), label=method2.__name__)
    plt.legend(loc=3)
    plt.savefig(f"Comparison_vpos_difference_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion v Variance n={n}, runs={n_runs}")
    plt.plot(tr, r_v_variance, label=method1.__name__)
    plt.plot(te, e_v_variance, label=method2.__name__)
    plt.plot(tr, vVarHarmonic(sig, m, w, tr), label="Analytic")
    plt.legend(loc=3)
    plt.savefig(f"Comparison_vvar_{n}_{n_runs}_" + add)
    # plt.show()

    plt.figure()
    plt.title(f"Comparsion v Variance Difference n={n}, runs={n_runs}")
    plt.plot(tr, r_v_variance - vVarHarmonic(sig, m, w, tr), label=method1.__name__)
    plt.plot(tr, e_v_variance - vVarHarmonic(sig, m, w, tr), label=method2.__name__)
    plt.legend(loc=3)
    plt.savefig(f"Comparison_vvar_difference_{n}_{n_runs}_" + add)
    # plt.show()


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


def manyRunVarianceComparison(n_runs, variables, method):
    n, dt, x0, v0, w, sig, m = variables
    r_x_average_pre = np.zeros([n_runs, n])

    r_x_average_sum = np.zeros(n)
    r_x_square_sum = np.zeros(n)

    for n_run in range(n_runs):
        if n_run % 100 == 0:
            print(f"Run: {n_run}")
        noise = genNoise(n, dt)
        tr, xr, vr = method(n, dt, x0, v0, w ** 2, sig / m, noise=noise)
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


def averageRunsData(n_runs, variables, add="", euler=False, rk4=False):
    if euler:
        if add != "":
            add = "_" + str(add)
        tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, eulerStocastic)
        np.savez_compressed(f"Dat/eu_{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                            xv=xv)
        np.savez(f"Dat/eu_{n_runs}_variables{add}", variables)

    if rk4:
        print("Doing RK4")
        if add != "":
            add = "_" + str(add)
        tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, rk4Stocastic)
        np.savez_compressed(f"Dat/rk4_{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                            xv=xv)
        np.savez(f"Dat/rk4_{n_runs}_variables{add}", variables)
    else:
        if add != "":
            add = "_" + str(add)
        tr, x_sum, x_square, v_sum, v_square, xv = averageRuns(n_runs, variables, rk2Stocastic)
        np.savez_compressed(f"Dat/rk_{n_runs}{add}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                            xv=xv)
        np.savez(f"Dat/rk_{n_runs}_variables{add}", variables)


def loadData(n_runs, plot=False, rk4=False):
    add = "rk"
    if rk4:
        add = "rk4"

    dat = np.load(f"Dat/{add}_{n_runs}_variables.npz")
    n, dt, x0, v0, w, sig, m = dat["arr_0"]
    var_list = [n, dt, x0, v0, w, sig, m]

    dat = np.load(f"Dat/{add}_{n_runs}.npz")
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
        plotData(x_average, v_average, x_variance, v_variance, sim_energy, var_list)
    return tl, x_average, v_average, x_variance, v_variance, sim_energy, var_list


def loadDataList(n_list, rk4=False):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Difference n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, rk4=rk4)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, x_average - xHarmonic(x0, v0, w, tl), label=f"{n_runs}")
        ax2.plot(tl, v_average - vHarmonic(x0, v0, w, tl))

    ax1.legend(loc=3)
    ax1.set_ylabel("x-x_expected")
    ax2.set_ylabel("v-v_expected")
    ax2.set_xlabel("t")
    plt.savefig("Spring_Diff")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("x variance n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, rk4=rk4)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, x_variance, label=f"{n_runs}")
        ax2.plot(tl, x_variance - xVarHarmonic(sig, m, w, tl))

    ax1.plot(tl, xVarHarmonic(sig, m, w, tl), label="Expected", color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("x var")
    ax2.set_ylabel("x var-expected")
    ax2.set_xlabel("t")
    plt.savefig("Spring_var")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("v variance n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, rk4=rk4)
        n, dt, x0, v0, w, sig, m = variables
        ax1.plot(tl, v_variance, label=f"{n_runs}")
        ax2.plot(tl, v_variance - vVarHarmonic(sig, m, w, tl))

    ax1.plot(tl, vVarHarmonic(sig, m, w, tl), label="Expected", color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("v var")
    ax2.set_ylabel("v var-expected")
    ax2.set_xlabel("t")
    plt.savefig("Spring_var_v")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Energy n")
    for n_runs in n_list:
        tl, x_average, v_average, x_variance, v_variance, sim_energy, variables = loadData(n_runs, rk4=rk4)
        ax1.plot(tl, sim_energy, label=f"{n_runs}")
        ax2.plot(tl, sim_energy - energyExpect(variables))

    ax1.plot(tl, energyExpect(variables), label="Expected", color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Difference")
    ax2.set_xlabel("t")
    plt.savefig("Spring_energ")
    plt.show()


def multipleData(rep, n_runs, variables, rk4):
    for i in range(rep):
        print(f"----------------------------------------------\nRepetition {i}")
        if rk4:
            print("Doing rk4")
            averageRunsData(n_runs, variables, add=i, rk4=True)
        else:
            averageRunsData(n_runs, variables, add=i)


def mergeData(rep, n_runs, n):
    x_sum = np.zeros(n)
    x_square = np.zeros(n)
    v_sum = np.zeros(n)
    v_square = np.zeros(n)
    xv = np.zeros(n)
    for i in range(rep):
        dat = np.load(f"Dat/rk4_{n_runs}_{i}.npz")
        x_sum += dat["x_sum"]
        x_square += dat["x_square"]
        v_sum += dat["v_sum"]
        v_square += dat["v_square"]
        xv = +dat["xv"]

    dat = np.load(f"Dat/rk4_{n_runs}_variables_0.npz")
    variables = dat["arr_0"]

    np.savez_compressed(f"Dat/rk4_{n_runs * rep}", x_sum=x_sum, x_square=x_square, v_sum=v_sum, v_square=v_square,
                        xv=xv)
    np.savez(f"Dat/rk4_{n_runs * rep}_variables", variables)


def plotData(x_average, v_average, x_variance, v_variance, sim_energy, variables):
    plt.plot(tl, x_average)
    plt.show()
    plt.plot(tl, v_variance)
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

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(tr, x_average)
    ax2.plot(tr, v_average)
    plt.show()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(tr, x_variance)
    ax1.plot(tr, xVarHarmonic(sig, m, w, tr))
    ax2.plot(tr, v_variance)
    # plt.savefig(method.__name__ + "_Variance_BiggestRuns")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(tl, sim_energy, label=f"{n_runs}")
    ax2.plot(tl, sim_energy - energyExpect(variables))
    plt.show()
    # plt.plot(tl, 0.5 * m * v_square / n_runs)
    # plt.plot(tl, 0.5 * m * w ** 2 * x_square / n_runs)
    # plt.show()
    # plt.plot(tl, 0.5 * m * v_square / n_runs + 0.5 * m * w ** 2 * x_square / n_runs)
    # plt.show()


def dtTesting(n_list, final_t, runs, variables, euler=False, rk4=False):
    n, dt, x0, v0, w, sig, m = variables
    for n_v in n_list:
        print(n_v)
        dt_v = final_t / n_v
        var_list_v = [n_v, dt_v, x0, v0, w, sig, m]
        averageRunsData(runs, var_list_v, add=dt_v, euler=euler, rk4=rk4)


def dtLoadData(n_runs, variables, plot=False, euler=False, rk4=False):
    n_v, dt_v, x0, v0, w, sig, m = variables
    if euler:
        dat = np.load(f"Dat/eu_{n_runs}_{dt_v}.npz")
    elif rk4:
        dat = np.load(f"Dat/rk4_{n_runs}_{dt_v}.npz")
    else:
        dat = np.load(f"Dat/rk_{n_runs}_{dt_v}.npz")
    x_sum = dat["x_sum"]
    x_square = dat["x_square"]
    v_sum = dat["v_sum"]
    v_square = dat["v_square"]
    xv = dat["xv"]

    tl = [i * dt_v for i in range(int(n_v))]

    x_average = average(x_sum, n_runs)
    v_average = average(v_sum, n_runs)
    x_variance = var(x_sum, x_square, n_runs)
    v_variance = var(v_sum, v_square, n_runs)
    sim_energy = energy(x_square, v_square, n_runs, m, w)
    if plot:
        plotData(x_average, v_average, x_variance, v_variance, sim_energy, var_list)
    return tl, x_average, v_average, x_variance, v_variance, sim_energy, var_list


def dtLoadList(n_list, final_t, runs, variables, euler=False, rk4=False):
    if euler:
        add = "_el"
    if rk4:
        add = "_rk4"
    else:
        add = ""
    n, dt, x0, v0, w, sig, m = variables
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Difference dt")
    for n_v in n_list:
        dt_v = final_t / n_v
        var_list_v = [n_v, dt_v, x0, v0, w, sig, m]
        tl, x_average, v_average, x_variance, v_variance, sim_energy, ignore_variables = dtLoadData(runs, var_list_v,
                                                                                                    euler=euler,
                                                                                                    rk4=rk4)
        ax1.plot(tl, abs(x_average - xHarmonic(x0, v0, w, tl)), label=f"{dt_v}")
        ax2.plot(tl, abs(v_average - vHarmonic(x0, v0, w, tl)))
        # ax1.plot(tl, x_average, label=f"{dt_v}")
        # ax2.plot(tl, v_average)
    ax1.legend(loc=4)
    ax1.set_ylabel("x-x_expected")
    ax2.set_ylabel("v-v_expected")
    ax2.set_xlabel("t")
    plt.savefig(f"Spring_Diff_dt{add}")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("x variance dt")
    for n_v in n_list:
        dt_v = final_t / n_v
        var_list_v = [n_v, dt_v, x0, v0, w, sig, m]
        tl, x_average, v_average, x_variance, v_variance, sim_energy, ignore_variables = dtLoadData(runs, var_list_v,
                                                                                                    euler=euler,
                                                                                                    rk4=rk4)
        ax1.plot(tl, x_variance, label=f"{dt_v}")
        ax2.plot(tl, abs(x_variance - xVarHarmonic(sig, m, w, tl)))

    ax1.plot(tl, xVarHarmonic(sig, m, w, tl), label="Expected", color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("x var")
    ax2.set_ylabel("x var-expected")
    ax2.set_xlabel("t")
    plt.savefig(f"Spring_var_dt{add}")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Energy dt")
    for n_v in n_list:
        dt_v = final_t / n_v
        var_list_v = [n_v, dt_v, x0, v0, w, sig, m]
        tl, x_average, v_average, x_variance, v_variance, sim_energy, ignore_variables = dtLoadData(runs, var_list_v,
                                                                                                    euler=euler,
                                                                                                    rk4=rk4)
        ax1.plot(tl, sim_energy, label=f"{dt_v}")
        ax2.plot(tl, abs(sim_energy - energyExpect(var_list_v)))

    ax1.plot(tl, energyExpect(var_list_v), label="Expected", color="k", linestyle="--")
    ax1.legend(loc=3)
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Difference")
    ax2.set_xlabel("t")
    plt.savefig(f"Spring_energ_dt{add}")
    plt.show()


d = 5
v = 1
t = d / v
print("Final time", t)
n = 5000
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

detectFolder()

n_runs = 10000

var_list = [n, dt, x0, v0, w, sig, m]
############ Basic Method Plotting
# averageRuns(n_runs, var_list, rk4Stocastic)
# plotAverageRuns(n_runs, var_list, rk4Stocastic)

# compareEulerRk2(var_list)
# compareEulerRk2ManyRun(n_runs, var_list)
# compareMethods(n_runs, var_list, rk4Stocastic, rk4Stocastic2, add="rk4")
# compareMethods(n_runs, var_list, rk2Stocastic, rk4Stocastic2, add="rk24")

# manyRunVarianceComparison(n_runs, var_list)


###### Data Gathering
# n_runs = 100
# averageRunsData(n_runs, var_list)
# loadData(n_runs, plot=True)

# n_runs_list = [100, 1000, 10000, 50000]
# for n_runs in n_runs_list:
#     averageRunsData(n_runs, var_list, rk4=True)

# n_runs_list = [10000, 50000, 100000]
# loadDataList(n_runs_list)
# loadDataList(n_runs_list, rk4=True)

# n_runs = 10000
# rep = 10
# multipleData(rep, n_runs, var_list, rk4=True)
# mergeData(rep, n_runs, n)

########## Dt Testing
n_runs = 10000
n_list = [500, 1000, 5000, 10000]

# dtTesting(n_list, t, n_runs, var_list, rk4=True)

# dtLoadList(n_list, t, n_runs, var_list)
n_list = [1000, 5000, 10000, 20000]
dtLoadList(n_list, t, n_runs, var_list, rk4=True)
