import numpy as np
import matplotlib.pyplot as plt


def multiplyDt(dt, n=0):
    return np.sqrt(dt) * np.random.randn(n)


def scaledDt(dt, n=0):
    return np.random.normal(size=n, loc=0, scale=np.sqrt(dt))


def testWeinerDist(dt, n):
    bins = 100

    multip_dt = multiplyDt(dt, n=n)
    scale_dt = scaledDt(dt, n)
    print("Output")
    print(f"Mean:{0}\tStd:{np.sqrt(dt)}")
    print(f"Multiply Method\tMean:{np.mean(multip_dt)}\tStd:{np.std(multip_dt)}")
    print(f"Scaling Method Method\tMean:{np.mean(scale_dt)}\tStd:{np.std(scale_dt)}")
    print("Difference")
    print(f"Multiply Method\tMean:{np.mean(multip_dt)}\tStd:{np.std(multip_dt) - np.sqrt(dt)}")
    print(f"Scaling Method Method\tMean:{np.mean(scale_dt)}\tStd:{np.std(scale_dt) - np.sqrt(dt)}")

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title("Multiplication Method")
    ax1.hist(multip_dt, bins=bins, density=True)
    ax2.set_title("Scaling Method")
    ax2.hist(scale_dt, bins=bins, density=True)
    plt.show()


def aRK(t, x, a):
    return -a * x


def eulerMethod(n, dt, x0, v0, a):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = v0
    for w in range(n - 1):
        x += vol * dt
        xl.append(x)

        v += -a * xol * dt
        vl.append(v)

        t += dt
        tl.append(t)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def rk4Method(n, dt, x0, v0, a):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = v0
    for w in range(n - 1):
        k0 = vol * dt
        l0 = -a * xol * dt
        k1 = (vol + l0 / 2) * dt
        l1 = -a * (xol + k0 / 2) * dt
        k2 = (vol + l1 / 2) * dt
        l2 = -a * (xol + k1 / 2) * dt
        k3 = (vol + l2) * dt
        l3 = -a * (xol + k2) * dt

        x += (k0 + 2 * k2 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6

        t += dt
        tl.append(t)
        xl.append(x)
        vl.append(v)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def f(t, x, v):
    return v


def g(t, x, v, a):
    return -a * x


def rk4Alt(n, dt, x0, v0, a):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    for w in range(n - 1):
        k0 = dt * f(t, x, v)
        l0 = dt * g(t, x, v, a)
        k1 = dt * f(t + dt / 2, x + k0 / 2, v + l0 / 2)
        l1 = dt * g(t + dt / 2, x + k0 / 2, v + l0 / 2, a)
        k2 = dt * f(t + dt / 2, x + k1 / 2, v + l1 / 2)
        l2 = dt * g(t + dt / 2, x + k1 / 2, v + l1 / 2, a)
        k3 = dt * f(t + dt, x + k2, v + l2)
        l3 = dt * g(t + dt, x + k2, v + l2, a)

        x += (k0 + 2 * k1 + 2 * k2 + k3) / 6
        v += (l0 + 2 * l1 + 2 * l2 + l3) / 6
        t += dt

        xl.append(x)
        vl.append(v)
        tl.append(t)
    return tl, np.asarray(xl), np.asarray(vl)


def rk2Method(n, dt, x0, v0, a):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = v0
    for w in range(n - 1):
        x += dt * (vol + -a * xol * dt / 2)
        xl.append(x)
        v += dt * (-a * (xol + vol * dt / 2))
        vl.append(v)

        t += dt
        tl.append(t)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def xHarmonic(x0, v0, w, t):
    t = np.asarray(t)
    return x0 * np.cos(w * t) + v0 * np.sin(w * t) / w


def vHarmonic(x0, v0, w, t):
    t = np.asarray(t)
    return -w * x0 * np.sin(w * t) + v0 * np.sin(w * t) / w ** 2


def energy(x, v, w, m):
    v = np.asarray(v)
    x = np.asarray(x)
    return 0.5 * v ** 2 + 0.5 * m * w ** 2 * x ** 2


def eHarmonic(x0, v0, w, t):
    x = xHarmonic(x0, v0, w, t)
    v = vHarmonic(x0, v0, w, t)
    return energy(x, v, w, m)


def solPlots(variables, t_lists, x_lists, v_lists, labels=None, title=None):
    ##### First element of t_list is used for comparison functions
    if labels == None:
        labels = [None for i in range(len(x_lists))]

    x0, v0, w, m = variables
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    for ind, x_run in enumerate(x_lists):
        ax1.plot(t_lists[ind + 1], x_run, label=labels[ind])
    ax1.plot(t_lists[0], xHarmonic(x0, v0, w, t_lists[0]), linestyle="--", color="k", label="Sol")
    ax1.legend(loc=3)

    for ind, v_run in enumerate(v_lists):
        ax2.plot(t_lists[ind + 1], v_run)
    ax2.plot(t_lists[0], vHarmonic(x0, v0, w, t_lists[0]), linestyle="--", color="k", label="Sol")

    ax1.set_ylabel("x")
    ax2.set_ylabel("v")
    ax2.set_xlabel("t")

    if title != None:
        fig.suptitle(title)
        plt.savefig("Numerics_" + title)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for ind, x_run in enumerate(x_lists):
        ax1.plot(t_lists[ind + 1], xHarmonic(x0, v0, w, t_lists[ind + 1]) - x_run, label=labels[ind])

    for ind, v_run in enumerate(v_lists):
        ax2.plot(t_lists[ind + 1], vHarmonic(x0, v0, w, t_lists[ind + 1]) - v_run)
    ax1.legend(loc=3)
    ax1.set_ylabel("x")
    ax2.set_ylabel("v")
    ax2.set_xlabel("t")

    if title != None:
        fig.suptitle(title + " Difference")
        plt.savefig("Numerics_" + title + "_Difference")
    plt.show()


def multipleRuns(final_time, n_list, variables, function, title=None):
    x0, v0, w, m = variables
    t_list = []
    x_list = []
    v_list = []
    for n in n_list:
        dt = final_time / n
        t_method, x_method, v_method = function(n, dt, x0, v0, w ** 2)
        t_list.append(t_method)
        x_list.append(x_method)
        v_list.append(v_method)

    t_list.insert(0, t_method)
    solPlots(variables, t_list, x_list, v_list, labels=[str(i) for i in n_list], title=title)

    return 0


def energyComparison(final_time, n, variables):
    x0, v0, w, m = variables
    dt = final_time / n
    t_e, x_e, v_e = eulerMethod(n, dt, x0, v0, w ** 2)
    e_e = energy(x_e, v_e, w, m)

    t_r, x_r, v_r = rk2Method(n, dt, x0, v0, w ** 2)
    e_r = energy(x_r, v_r, w, m)

    t_f, x_f, v_f = rk4Method(n, dt, x0, v0, w ** 2)
    e_f = energy(x_f, v_f, w, m)

    t_fa, x_fa, v_fa = rk4Alt(n, dt, x0, v0, w ** 2)
    e_fa = energy(x_fa, v_fa, w, m)

    e_expect = eHarmonic(x0, v0, w, t_e)
    plt.plot(t_e, (e_e - e_expect) / e_expect, label="Euler")
    plt.plot(t_r, (e_r - e_expect) / e_expect, label="Rk2")
    plt.plot(t_f, (e_f - e_expect) / e_expect, label="Rk4")
    plt.legend()
    plt.title("Energy discrepancy comparison")
    plt.xlabel("Time")
    plt.ylabel("dE/E")
    plt.show()

    plt.figure()
    # plt.plot(t_r, (e_r - e_expect) / e_expect, label="Rk2")
    plt.plot(t_f, (e_f - e_expect) / e_expect, label="Rk4")
    plt.plot(t_f, (e_fa - e_expect) / e_expect, label="Rk4Alt")
    # plt.plot(t_f, (e_r-e_f))
    plt.legend()
    plt.title("Energy discrepancy comparison Runge")
    plt.xlabel("Time")
    plt.ylabel("dE/E")
    plt.show()


def averageRuns(n_runs, variables):
    x0, v0, w, m = variables
    x_average = np.zeros(n)
    x_variance = np.zeros(n)
    v_average = np.zeros(n)
    v_variance = np.zeros(n)
    for n_run in range(n_runs):
        if n_run % 100 == 0:
            print(f"Run: {n_run}")
        tr, xr, vr = rk2Method(n, dt, x0, v0, w ** 2)
        if n_run == 0:
            x_average = xr
            v_average = vr
        else:
            x_average += (xr - x_average) / n_runs
            x_variance += (xr - x_average) * (xr - x_average)

            v_average += (vr - v_average) / n_runs
            v_variance += (vr - v_average) * (vr - v_average)
    return tr, x_average, x_variance, v_average, v_variance


n = 10000
final_time = 5
dt = final_time / n
plt.show()

n = 5000
dt = 0.001
final_time = 5
t_gen = np.arange(0, n * dt, dt)

w = 5
x0 = 10
v0 = 0
m = 1
var_list = [x0, v0, w, m]

# multipleRuns(final_time, [5000, 10000, 50000], var_list, eulerMethod, title="Euler Method")
energyComparison(final_time, n, var_list)
# tr, x_average, x_variance, v_average, v_variance = averageRuns(100,var_list)
# plt.plot(tr,x_average)
# plt.show()
# multipleRuns(final_time, [5000, 10000, 50000], var_list, rk4Method, title="Runge-Kuta 4")
# energyComparison(final_time, n, var_list)
