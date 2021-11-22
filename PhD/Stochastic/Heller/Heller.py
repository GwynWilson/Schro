import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Heller():
    def __init__(self, n, dt, init, func):
        """
        :param n:
        :param dt:
        :param init: Array of inital parmeters [x0,v0,a0,g0]
        :param func: Derivatives to be integrated
        """
        self.n = n
        self.dt = dt
        self.init = init

        self.derivatives = func
        self.args = None
        self.noise = None

        self.tl = None
        self.xl = None
        self.vl = None
        self.al = None
        self.gl = None

        self.x_av = np.zeros(self.n)
        self.v_av = np.zeros(self.n)
        self.a_av = np.zeros(self.n)
        self.g_av = np.zeros(self.n)

        self.xs = None
        self.vs = None
        self.xs_av = np.zeros(self.n)
        self.vs_av = np.zeros(self.n)

    def set_derivatives(self, func):
        self.derivatives = func

    def genNoise(self):
        # return np.sqrt(self.dt) * np.random.randn(self.n)

        return np.random.randn(self.n) / np.sqrt(self.dt)

    def genNoisedt(self):
        return np.sqrt(self.dt) * np.random.randn(self.n)

    def euler(self, args, noise=None):
        xl = [self.init[0]]
        vl = [self.init[1]]
        al = [self.init[2]]
        gl = [self.init[3]]
        t = 0
        tl = [0]
        current = np.asarray(self.init, dtype=complex)

        if noise == None:
            noise = self.genNoise()

        for i in range(self.n - 1):
            eta = noise[i]
            # eta = np.random.randn()
            values = self.dt * self.derivatives(t, current, args, eta, self.dt)
            current += np.asarray(values)
            t += self.dt

            xl.append(current[0])
            vl.append(current[1])
            al.append(current[2])
            gl.append(current[3])
            tl.append(t)

        return tl, np.asarray(xl), np.asarray(vl), np.asarray(al), np.asarray(gl)

    def rk4(self, args, noise=None, set=True, square=False):
        """
        :param args: Array of arguments for the derivatives function
        :param noise: Given noise to use, if not specified genereates white noise
        :param set: Saves the integrated values for x,v....
        :return:
        """
        self.noise = noise
        self.args = args
        xl = [self.init[0]]
        vl = [self.init[1]]
        al = [self.init[2]]
        gl = [self.init[3]]
        t = 0
        tl = [0]
        current = np.asarray(self.init, dtype=complex)

        if noise is None:
            self.noise = self.genNoise()

        if square:
            xs = [self.init[0] ** 2]
            vs = [self.init[1] ** 2]

        for i in range(self.n - 1):
            N = self.noise[i]
            # eta = np.random.randn()
            k0 = self.dt * np.asarray(self.derivatives(t, current, self.args, N, self.dt))
            k1 = self.dt * np.asarray(self.derivatives(t + self.dt / 2, current + k0 / 2, self.args, N, self.dt))
            k2 = self.dt * np.asarray(self.derivatives(t + self.dt / 2, current + k1 / 2, self.args, N, self.dt))
            k3 = self.dt * np.asarray(self.derivatives(t + self.dt, current + k2, args, N, self.dt))

            t += self.dt
            for j in range(4):
                current[j] += (k0[j] + 2 * k1[j] + 2 * k2[j] + k3[j]) / 6

            xl.append(current[0])
            vl.append(current[1])
            al.append(current[2])
            gl.append(current[3])
            tl.append(t)

            if square:
                xs.append(current[0] ** 2)
                vs.append(current[1] ** 2)

        if set:
            self.tl = np.asarray(tl)
            self.xl = np.asarray(xl)
            self.vl = np.asarray(vl)
            self.al = np.asarray(al)
            self.gl = np.asarray(gl)

        if square:
            self.xs = np.asarray(xs)
            self.vs = np.asarray(vs)

        return tl, np.asarray(xl), np.asarray(vl), np.asarray(al), np.asarray(gl)

    def rk4dt(self, args, noise=None, set=True, square=False):
        """
        :param args: Array of arguments for the derivatives function
        :param noise: Given noise to use, if not specified genereates white noise
        :param set: Saves the integrated values for x,v....
        :return:
        """
        self.noise = noise
        self.args = args
        xl = [self.init[0]]
        vl = [self.init[1]]
        al = [self.init[2]]
        gl = [self.init[3]]
        t = 0
        tl = [0]
        current = np.asarray(self.init, dtype=complex)

        if noise is None:
            self.noise = self.genNoisedt()

        if square:
            xs = [self.init[0] ** 2]
            vs = [self.init[1] ** 2]

        for i in range(self.n - 1):
            N = self.noise[i]
            # eta = np.random.randn()
            k0 = np.asarray(self.derivatives(t, current, self.args, N, self.dt))
            k1 = np.asarray(self.derivatives(t + self.dt / 2, current + k0 / 2, self.args, N, self.dt))
            k2 = np.asarray(self.derivatives(t + self.dt / 2, current + k1 / 2, self.args, N, self.dt))
            k3 = np.asarray(self.derivatives(t + self.dt, current + k2, args, N, self.dt))

            t += self.dt
            for j in range(4):
                current[j] += (k0[j] + 2 * k1[j] + 2 * k2[j] + k3[j]) / 6

            xl.append(current[0])
            vl.append(current[1])
            al.append(current[2])
            gl.append(current[3])
            tl.append(t)

            if square:
                xs.append(current[0] ** 2)
                vs.append(current[1] ** 2)

        if set:
            self.tl = np.asarray(tl)
            self.xl = np.asarray(xl)
            self.vl = np.asarray(vl)
            self.al = np.asarray(al)
            self.gl = np.asarray(gl)

        if square:
            self.xs = np.asarray(xs)
            self.vs = np.asarray(vs)

        return tl, np.asarray(xl), np.asarray(vl), np.asarray(al), np.asarray(gl)

    def averageRuns(self, n_runs, args, euler=False, noise=None, square=False, dtver=False, save=""):
        self.x_av = np.zeros(self.n, dtype=complex)
        self.v_av = np.zeros(self.n, dtype=complex)
        self.a_av = np.zeros(self.n, dtype=complex)
        self.g_av = np.zeros(self.n, dtype=complex)

        if square:
            self.xs_av = np.zeros(self.n, dtype=complex)
            self.vs_av = np.zeros(self.n, dtype=complex)

        for i in range(n_runs):
            if euler:
                t_temp, x_temp, v_temp, a_temp, g_temp = self.euler(args)

            elif dtver:
                t_temp, x_temp, v_temp, a_temp, g_temp = self.rk4dt(args, noise=noise, set=False, square=square)

            else:
                t_temp, x_temp, v_temp, a_temp, g_temp = self.rk4(args, noise=noise, set=False, square=square)
            self.x_av += x_temp / n_runs
            self.v_av += v_temp / n_runs
            self.a_av += a_temp / n_runs
            self.g_av += g_temp / n_runs

            if square:
                self.xs_av += self.xs / n_runs
                self.vs_av += self.vs / n_runs

        self.tl = t_temp

        if save != "":
            np.savez_compressed(f"{save}_{n_runs}",tl=self.tl,x_av=self.x_av,v_av=self.v_av,a_av=self.a_av,g_av=self.g_av)

        return 0

    def plotBasic(self, expected=None, average=False, title="", noshow=False):
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
        if title == "":
            fig.suptitle(f"Heller Simulation")
        else:
            fig.suptitle(title)

        if average:
            axs[0, 0].plot(self.tl, self.x_av)
            axs[0, 1].plot(self.tl, self.v_av)
            axs[1, 0].plot(self.tl, np.imag(self.a_av))
            axs[1, 1].plot(self.tl, self.g_av)

        else:
            axs[0, 0].plot(self.tl, self.xl)
            axs[0, 1].plot(self.tl, self.vl)
            axs[1, 0].plot(self.tl, np.imag(self.al))
            axs[1, 1].plot(self.tl, self.gl)

        axs[0, 0].set_title("X Pos")
        axs[0, 0].set_ylabel("x")
        axs[0, 1].set_title("V Pos")
        axs[0, 1].set_ylabel("v")
        axs[1, 0].set_title(r"Alpha (Imaginary Part)")
        axs[1, 0].set_ylabel(r"$\alpha$ (Imaginary Part)")
        axs[1, 0].set_xlabel(r"t")
        axs[1, 1].set_title("Gamma")
        axs[1, 1].set_ylabel(r"$\gamma$")
        axs[1, 1].set_xlabel(r"t")
        if expected != None:
            x_ex, v_ex, a_ex, g_ex = expected(self.tl, self.args, self.init)
            axs[0, 0].plot(self.tl, x_ex, linestyle="--")
            axs[0, 1].plot(self.tl, v_ex, linestyle="--")
            axs[1, 0].plot(self.tl, np.imag(a_ex), linestyle="--")
            axs[1, 1].plot(self.tl, g_ex, linestyle="--")

        plt.show()
        return 0

    def plotDiffBasic(self, expected, average=False, title=""):
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
        if title == "":
            fig.suptitle(f"Heller Simulation")
        else:
            fig.suptitle(title)
        x_ex, v_ex, a_ex, g_ex = expected(self.tl, self.args, self.init)

        if average:
            axs[0, 0].plot(self.tl, x_ex - self.x_av, linestyle="--")
            axs[0, 1].plot(self.tl, v_ex - self.v_av, linestyle="--")
            axs[1, 0].plot(self.tl, np.imag(a_ex - self.a_av), linestyle="--")
            axs[1, 1].plot(self.tl, g_ex - self.g_av, linestyle="--")

        else:
            axs[0, 0].plot(self.tl, x_ex - self.xl, linestyle="--")
            axs[0, 1].plot(self.tl, v_ex - self.vl, linestyle="--")
            axs[1, 0].plot(self.tl, np.imag(a_ex - self.al), linestyle="--")
            axs[1, 1].plot(self.tl, g_ex - self.gl, linestyle="--")

        axs[0, 0].set_title("X Pos")
        axs[0, 1].set_title("V Pos")
        axs[1, 0].set_title("Alpha (Imaginary Part)")
        axs[1, 1].set_title("Gamma")
        plt.show()

    def plotPhase(self):
        plt.plot(self.xl, self.vl)
        plt.show()

    def plotSquare(self, expected=None, title=""):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
        if title == "":
            fig.suptitle(f"Heller Squared")
        else:
            fig.suptitle(title)

        axs[0].plot(self.tl, self.xs_av)
        axs[1].plot(self.tl, self.vs_av)

        if expected != None:
            xs_ex, vs_ex = expected(self.tl, self.args, self.init)
            axs[0].plot(self.tl, xs_ex, linestyle=":")
            axs[1].plot(self.tl, vs_ex, linestyle=":")

        axs[0].set_ylabel(r"$\langle x^2 \rangle$")
        axs[1].set_ylabel(r"$\langle v^2 \rangle$")
        axs[1].set_xlabel("Time (s)")
        plt.show()

    def getState(self, index):
        return self.xl[index], self.vl[index], self.al[index], self.gl[index]

    def getValues(self):
        return self.tl, self.xl, self.vl, self.al, self.gl

    def getAverage(self):
        return self.tl, self.x_av, self.v_av, self.a_av, self.g_av


def derivs(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1]
    v = -w ** 2 * current[0]
    a = (-2 * current[2] ** 2 / m) - (m * w ** 2) / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * current[0] ** 2 / 2
    return x, v, a, g


def derivsStoc(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1]
    v = -w ** 2 * (current[0] - eta * sig)
    a = (-2 * current[2] ** 2 / m) - (m * w ** 2) / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * (current[0] - eta * sig) ** 2 / 2
    return x, v, a, g


def derivsStocdt(t, current, args, eta, dt):
    w, m, hbar, sig = args
    x = current[1] * dt
    v = -w ** 2 * (current[0] * dt - eta * sig)
    a = ((-2 * current[2] ** 2 / m) - (m * w ** 2) / 2) * dt
    g = (1j * hbar * current[2] / m + m * current[1] ** 2 / 2) * dt - 0.5 * m * w ** 2 * current[
        0] ** 2 * dt + m * w ** 2 * current[0] * eta * sig - 0.5 * m * w ** 2 * sig ** 2 * eta ** 2
    return x, v, a, g


def expected(t, args, init):
    t = np.asarray(t)
    w, m, hbar, sig = args
    x0, v0, a0, g0 = init
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t)
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))
    g_ex = g0 - hbar * w * t / 2 + 0.5 * m * (v_ex * x_ex - v0 * x0)
    return x_ex, v_ex, a_ex, g_ex


def expected2(t, args, init):
    t = np.asarray(t)
    w, m, hbar, sig = args
    x0, v0, a0, g0 = init
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t)
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))
    g_ex = g0 - hbar * w * t / 2 + 0.5 * m * (v_ex * x_ex - v0 * x0) - m * sig ** 2 * t / (2 * w)
    return x_ex, v_ex, a_ex, g_ex


if __name__ == "__main__":
    n = 10000
    dt = 0.0001

    w = 10
    sig = 1
    m = 1
    hbar = 1
    args = (w, m, hbar, sig)
    temp = 0.5 * m * w ** 2

    a0 = 1j * m * w / 2
    init = [10, 5, a0, 0]
    # init = [0, 0, a0, 0]

    # solver = Heller(n, dt, init, derivs)
    # solver.rk4(args)
    # solver.plotBasic(expected=expected, title="Heller Simulation Non Stochastic")
    # solver.plotDiffBasic(expected)

    # solverStoc = Heller(n, dt, init, derivsStoc)
    # solverStoc.averageRuns(100, args)
    # solverStoc.plotBasic(expected=expected, average=True, title="Heller Simulation Stochastic")
    # solverStoc.plotDiffBasic(expected, average=True)

    # plt.plot(solverStoc.tl, solverStoc.g_av, label="Simulation")
    # plt.plot(solverStoc.tl, expected(solverStoc.tl, args, init)[3], label="Non Stochastic", linestyle="--")
    # plt.plot(solverStoc.tl, expected2(solverStoc.tl, args, init)[3], label="Theory", linestyle="--")
    # plt.legend()
    # plt.savefig("Heller Gamma")
    # plt.show()

    ################ dt Testing
    nruns = 10
    solver = Heller(n, dt, init, derivsStocdt)
    solver.averageRuns(nruns, args, dtver=True)
    solver.plotBasic(average=True, expected=expected2, title=f"Stochastic Oscillator n={nruns}")
