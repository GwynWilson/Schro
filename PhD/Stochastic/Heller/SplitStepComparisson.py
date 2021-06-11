from Schrodinger_Solver import Schrodinger
from Heller import Heller
from scipy.fftpack import fftfreq, fftshift
from scipy.integrate import simps

import numpy as np
import matplotlib.pyplot as plt


class SplitComp():
    def __init__(self, sch, hel):
        self.sch = sch
        self.hel = hel

    def hellerPacket(self, xt, vt, at, gt):
        return np.exp(
            (1j / self.sch.hbar) * at * (self.sch.x - xt) ** 2 + (1j / self.sch.hbar) * self.sch.m * vt * (x - xt) + (
                    1j / self.sch.hbar) * gt)

    def hellerInitialPacket(self):
        xt, vt, at, gt = self.hel.init
        return np.exp(
            (1j / hbar) * at * (self.sch.x - xt) ** 2 + (1j / hbar) * self.sch.m * vt * (x - xt) + (1j / hbar) * gt)

    def hellerRun(self, args):
        self.hel.rk4(args)

    def packetComparisson(self, args, step=1, Nsteps=None):
        self.hel.rk4(args)

        if Nsteps == None:
            Nsteps = self.hel.n

        psi_h = self.hellerInitialPacket()
        # plt.plot(self.sch.x,psi_h)
        # plt.plot(self.sch.x,self.sch.psi_x)
        # plt.show()

        t_list = []
        squared_list = []
        norm_h = []
        norm_s = []
        for i in range(Nsteps):
            if i != 0:
                sch.evolve_t(step, self.hel.dt)
            t_list.append(self.sch.t)
            psi_sol = self.sch.psi_x
            xt, vt, at, gt = self.hel.getState(i)
            psi_h = self.hellerPacket(xt, vt, at, gt)
            # plt.plot(x, psi_h)
            # plt.plot(x, psi_x)
            # plt.show()
            squared = np.conjugate(psi_sol) * psi_h
            squared_list.append(simps(squared, sch.x))

            psi_hs = np.conjugate(psi_h) * psi_h
            norm_h.append(simps(psi_hs, self.sch.x))
            norm_s.append(self.sch.norm_x())

        # plt.plot(self.sch.x,psi_h)
        # plt.plot(self.sch.x,self.sch.psi_x)
        # plt.show()

        plt.title("States of Packet")
        psi_h = self.hellerPacket(self.hel.xl[-1], self.hel.vl[-1], self.hel.al[-1], self.hel.gl[-1])
        psi_h_init = self.hellerPacket(self.hel.xl[0], self.hel.vl[0], self.hel.al[0], self.hel.gl[0])
        plt.plot(self.sch.x, psi_h * np.conjugate(psi_h))
        plt.plot(self.sch.x, self.sch.mod_square_x(r=True))
        plt.plot(self.sch.x, psi_h_init * np.conjugate(psi_h_init))
        plt.show()

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(x, np.real(psi_h), label="sch")
        axs[0].plot(x, np.real(self.sch.psi_x), label="hel")
        axs[1].plot(x, np.imag(psi_h))
        axs[1].plot(x, np.imag(self.sch.psi_x))
        axs[0].legend()
        plt.show()

        # plt.plot(self.sch.x, self.sch.mod_square_x(r=True) - psi_h * np.conjugate(psi_h))
        # plt.show()

        plt.title("Normalisation")
        plt.plot(t_list,norm_h,label="heller")
        plt.plot(t_list,norm_s,label="Sch")
        plt.legend()
        plt.show()

        plt.title("overlap")
        plt.plot(t_list, squared_list)
        plt.show()

    def comparrisonKick(self, args, step=1, kick=0):
        Ntot = self.hel.n
        Nhalf = int(Ntot / 2)
        solver1 = Heller(Nhalf, self.hel.dt, self.hel.init, self.hel.derivatives)
        tl1, xl1, vl1, al1, gl1 = solver1.rk4(args)

        init2 = [xl1[-1], vl1[-1] + kick / self.sch.m, al1[-1], gl1[-1]]
        Solver2 = Heller(Nhalf, self.hel.dt, init2, self.hel.derivatives)
        tl2, xl2, vl2, al2, gl2 = Solver2.rk4(args)

        tlcomb = np.concatenate((tl1, np.asarray(tl2[1:]) + tl1[-1]), axis=None)
        xlcomb = np.concatenate((xl1, xl2[1:]), axis=None)
        vlcomb = np.concatenate((vl1, vl2[1:]), axis=None)
        alcomb = np.concatenate((al1, al2[1:]), axis=None)
        glcomb = np.concatenate((gl1, gl2[1:]), axis=None)

        self.hel.tl = tlcomb
        self.hel.xl = xlcomb
        self.hel.vl = vlcomb
        self.hel.al = alcomb
        self.hel.gl = glcomb

        self.hel.plotBasic()

        t_list = []
        squared_list = []
        norm_h = []
        norm_s = []
        for i in range(Nhalf):
            if i != 0:
                sch.evolve_t(step, self.hel.dt)
            t_list.append(self.sch.t)
            psi_sol = self.sch.psi_x
            xt, vt, at, gt = self.hel.getState(i)
            psi_h = self.hellerPacket(xt, vt, at, gt)
            squared = np.conjugate(psi_sol) * psi_h
            squared_list.append(simps(squared, self.sch.x))

            psi_hs = np.conjugate(psi_h) * psi_h
            norm_h.append(simps(psi_hs, self.sch.x) - 1)
            norm_s.append(self.sch.norm_x() - 1)

        self.sch.momentum_kick(kick / self.sch.hbar)

        for i in range(Nhalf - 1):
            sch.evolve_t(step, self.hel.dt)
            t_list.append(self.sch.t)
            psi_sol = self.sch.psi_x
            xt, vt, at, gt = self.hel.getState(Nhalf + i)
            psi_h = self.hellerPacket(xt, vt, at, gt)
            squared = np.conjugate(psi_sol) * psi_h
            # print(simps(squared, self.sch.x))
            squared_list.append(simps(squared, self.sch.x))

            psi_hs = np.conjugate(psi_h) * psi_h
            norm_h.append(simps(psi_hs, self.sch.x) - 1)
            norm_s.append(self.sch.norm_x() - 1)

        ####Final state
        psi_h = self.hellerPacket(self.hel.xl[-1], self.hel.vl[-1], self.hel.al[-1], self.hel.gl[-1])
        plt.plot(self.sch.x, psi_h * np.conjugate(psi_h))
        plt.plot(self.sch.x, self.sch.mod_square_x(r=True))
        plt.show()

        plt.plot(self.sch.x, self.sch.mod_square_x(r=True) - psi_h * np.conjugate(psi_h))
        plt.show()

        plt.plot(t_list, norm_h)
        plt.plot(t_list, norm_s)
        plt.show()

        plt.plot(t_list, np.real(np.asarray(squared_list)), label="Real")
        # plt.plot(t_list, np.imag(np.asarray(squared_list)), label="Imag")
        # plt.legend()
        plt.show()


if __name__ == "__main__":
    def gauss_init(x, k0, x0=0, d=1):
        return 1 / np.sqrt((d * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (4 * d ** 2)) * np.exp(
            1j * k0 * (x - x0))


    def hellerPacket(x, xt, vt, at, gt):
        return np.exp(
            (1j / hbar) * at * (x - xt) ** 2 + (1j / hbar) * m * vt * (x - xt) + (
                    1j / hbar) * gt)


    def derivs(t, current, args, eta, dt):
        m, hbar, = args
        x = current[1]
        v = 0
        a = (-2 * current[2] ** 2) / m
        g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2
        return x, v, a, g


    # Defining x axis
    N = 2 ** 10
    dx = 0.1
    x_length = N * dx
    x = np.asarray([i * dx for i in range(N)])
    x0 = int(0.25 * x_length)

    d = 1

    # Defining Psi and V
    k_initial = 2
    psi_x = gauss_init(x, k_initial, x0, d=d)
    V_x = np.zeros(N)

    # Defining K range
    dk = dx / (2 * np.pi)
    k = fftfreq(N, dk)
    ks = fftshift(k)

    # Defining time steps
    t = 0
    dt = 0.01
    step = 1
    Ns = 1000
    print("Final time", dt * step * Ns)

    hbar = 1
    m = 1

    args = (m, hbar)

    p0 = k_initial * hbar
    a0 = 1j * hbar / (4 * d ** 2)
    g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)
    init = [x0, p0 / m, a0, g0]

    psi_x = hellerPacket(x, x0, p0 / m, a0, g0)

    sch = Schrodinger(x, psi_x, V_x, k, hbar=hbar, m=m)

    hel = Heller(Ns, dt, init, derivs)

    comp = SplitComp(sch, hel)
    comp.packetComparisson(args)

    # comp.comparrisonKick(args, kick=0)
    # comp.comparrisonKick(args, kick=-2 * p0)
