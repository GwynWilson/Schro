import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def randGauss(mu, sig, n):
    return sig * np.random.randn(n) + mu


def weinerProcess(n, dt, w0=0):
    temp = []
    for i in range(n):
        if i != 0:
            w0 += norm.rvs(scale=2 * dt)
        temp.append(w0)
    return temp


def itoProcess(n, dt, a, b, x0=0, t0=0, w=None, aargs=None, bargs=None):
    t = t0
    x = x0
    temp = []
    if w == None:
        w = weinerProcess(n, dt)
        # w = np.sqrt(dt)*np.random.randn(n)
    for i in range(n):
        tempx = 0
        if i != 0:
            if callable(a):
                if aargs != None:
                    tempx += a(x, t, aargs) * dt
                else:
                    tempx += a(x, t) * dt
            else:
                tempx += a * dt
            if callable(b):
                if bargs != None:
                    tempx += b(x, t, bargs) * (w[i] - w[i - 1])
                else:
                    tempx += b(x, t) * (w[i] - w[i - 1])
            else:
                tempx += b * (w[i] - w[i - 1])
        x += tempx
        temp.append(x)
        t += dt
    return temp


def itoProcess2(n, dt, a, b, x0=0, t0=0, aargs=None, bargs=None):
    t = t0
    x = x0
    temp = [x0]
    for i in range(n-1):
        tempx = 0
        if callable(a):
            if aargs != None:
                tempx += a(x, t, aargs) * dt
            else:
                tempx += a(x, t) * dt
        else:
            tempx += a * dt

        dw = np.sqrt(dt)*np.random.randn()
        if callable(b):
            if bargs != None:
                tempx += b(x, t, bargs) * dw
            else:
                tempx += b(x, t) * dw
        else:
            tempx += b * dw
        x += tempx
        temp.append(x)
        t += dt
    return temp


def manyRuns(run, n, dt, a, b, x0=0, t0=0, w=None, aargs=None, bargs=None):
    # runs = np.zeros([run, n], dtype=complex)
    runs = np.zeros([run, n])
    for i in range(run):
        process = itoProcess2(n, dt, a, b, x0=x0, t0=t0, aargs=aargs, bargs=bargs)
        runs[i] = process
    return runs


def plotRuns(runs):
    for i in runs:
        plt.plot(i)
    plt.show()


def itoAverages(runs):
    nruns, nlength = runs.shape
    average = np.zeros([nlength], dtype=complex)
    for run in runs:
        for i, v in enumerate(run):
            average[i] += v / nruns
    return average


def runsData(data):
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        dslice = data[:, i]
        average[i] = np.mean(dslice)
        variance[i] = np.var(dslice)
    return average, variance


def test(x, t):
    return np.sin(t)


if __name__ == "__main__":
    n = 1000
    dt = 0.01
    t_list = [i * dt for i in range(n)]
    a = 1
    b = 0.1
    x0 = 0

    run = 15
    runs = np.zeros([run, n])
    for i in range(run):
        process = itoProcess2(n, dt, test, b)
        plt.plot(process)
        runs[i] = process
    # plt.savefig("Ito_run")
    plt.show()

    plt.plot(itoAverages(runs))
    plt.plot(-1 * np.cos(t_list) + np.cos(0))
    # plt.savefig("Ito average")
    plt.show()
