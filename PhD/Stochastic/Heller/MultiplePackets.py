import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PhD.Stochastic.Heller import Heller as hl
from Schrodinger_Solver import Schrodinger
from scipy.integrate import simps


class HellerInterference():

    def __init__(self, n, dt, initarray, func, args, xarr):
        self.initarray = np.copy(initarray)
        self.N_packet = len(initarray)

        self.n = n
        self.dt = dt
        self.derivs = func
        self.args = args
        self.m = args[0]
        self.hbar = args[1]
        self.x = xarr

    def hellerPacket(self, xt, vt, at, gt):
        return np.exp(
            (1j / self.hbar) * at * (self.x - xt) ** 2 + (1j / self.hbar) * self.m * vt * (self.x - xt) + (
                    1j / self.hbar) * gt)

    def runAll(self):
        psi_comb = np.zeros((self.n, len(self.x)), dtype=complex)
        for initi in self.initarray:
            tempHel = hl.Heller(self.n, self.dt, initi, self.derivs)
            tempHel.rk4(self.args)

            psi_arr = np.zeros((self.n, len(self.x)), dtype=complex)
            for i in range(self.n):
                xi, vi, ai, gi = tempHel.getState(i)
                psi_i = self.hellerPacket(xi, vi, ai, gi)
                psi_comb[i] += psi_i / np.sqrt(self.N_packet)
            # psi_comb += psi_arr / np.sqrt(self.N_packet)
            # psi_comb += psi_arr / self.N_packet
        self.psi_comb = psi_comb
        return self.psi_comb

    def onePacket(self, kick, init, plot=False, final=False, dat=False):
        Nhalf = int(self.n / 2)
        tempHel = hl.Heller(Nhalf, self.dt, init, self.derivs)
        tl, xl, vl, al, gl = tempHel.rk4(self.args)

        psi_arr = np.zeros((self.n, len(self.x)), dtype=complex)
        for i in range(Nhalf):
            xi, vi, ai, gi = tempHel.getState(i)
            psi_i = self.hellerPacket(xi, vi, ai, gi)
            psi_arr[i] = psi_i

        init2 = [xl[-1], vl[-1] + kick, al[-1], gl[-1]]
        tempHel2 = hl.Heller(Nhalf, self.dt, init2, self.derivs)
        tl2, xl2, vl2, al2, gl2 = tempHel2.rk4(self.args)
        for i in range(Nhalf):
            xi, vi, ai, gi = tempHel2.getState(i)
            psi_i = self.hellerPacket(xi, vi, ai, gi)
            psi_arr[i + Nhalf] = psi_i

        if plot:
            tlcomb = np.concatenate((tl, np.asarray(tl2[1:]) + tl[-1]), axis=None)
            xlcomb = np.concatenate((xl, xl2[1:]), axis=None)
            vlcomb = np.concatenate((vl, vl2[1:]), axis=None)
            alcomb = np.concatenate((al, al2[1:]), axis=None)
            glcomb = np.concatenate((gl, gl2[1:]), axis=None)

            tempHel.tl = tlcomb
            tempHel.xl = xlcomb
            tempHel.vl = vlcomb
            tempHel.al = alcomb
            tempHel.gl = glcomb

            tempHel.plotBasic()

        if final:
            psi_f = self.hellerPacket(xl2[-1], vl2[-1] - kick / 2, al2[-1], gl2[-1])
            return psi_arr, psi_f
        elif dat:
            return psi_arr, tlcomb, xlcomb, vlcomb, alcomb, glcomb

        else:
            return psi_arr

    def interferometry(self):
        psi_comb = np.zeros((self.n, len(self.x)), dtype=complex)
        for initi in self.initarray:
            psi_arr = self.onePacket(-2 * initi[1], initi, plot=True)

            psi_comb += psi_arr / np.sqrt(self.N_packet)
        self.psi_comb = psi_comb
        return self.psi_comb

    def interferometry2(self, kick, plot=False, psii=False, psif=False):
        psi_comb = np.zeros((self.n, len(self.x)), dtype=complex)

        init1 = self.initarray[0]
        init2 = self.initarray[1]

        if psii:
            psi_1 = self.hellerPacket(init1[0], init1[1], init1[2], init1[3])
            psi_2 = self.hellerPacket(init2[0], init2[1], init2[2], init2[3])
            psi_init = 1 / np.sqrt(self.N_packet) * (psi_1 + psi_2)

        init1[1] = kick
        init2[1] = -kick

        if psif:
            psi_arr1, psi_final_1 = self.onePacket(-2 * kick, init1, plot=plot, final=psif)
            psi_arr2, psi_final_2 = self.onePacket(2 * kick, init2, plot=plot, final=psif)
            psi_final = 1 / np.sqrt(self.N_packet) * (psi_final_1 + psi_final_2)

        else:
            psi_arr1 = self.onePacket(-2 * kick, init1, plot=plot)
            psi_arr2 = self.onePacket(2 * kick, init2, plot=plot)

        psi_comb = (psi_arr1 + psi_arr2) / np.sqrt(self.N_packet)

        if psif:
            psi_comb = np.append(psi_comb, [psi_final], axis=0)
        if psii:
            psi_comb = np.insert(psi_comb, 0, psi_init, axis=0)
        self.psi_comb = psi_comb

        return self.psi_comb

    def modSquare(self):
        shape = np.shape(self.psi_comb)
        self.psi_squared = np.zeros(shape)

        for i, v in enumerate(self.psi_comb):
            self.psi_squared[i] = np.real(np.conjugate(v) * v)
        return self.psi_squared


class SchroInterference():

    def __init__(self, n, dt, x, v, initarray, hbar=1, m=1):
        self.n = n
        self.dt = dt

        self.x = x
        self.hbar = hbar
        self.m = m
        self.v = v

        self.initarray = np.copy(initarray)
        self.N_packet = len(initarray)

    def hellerPacket(self, xt, vt, at, gt):
        return np.exp(
            (1j / self.hbar) * at * (self.x - xt) ** 2 + (1j / self.hbar) * self.m * vt * (self.x - xt) + (
                    1j / self.hbar) * gt)

    def runAll(self, N_step=1):
        psi_comb = np.zeros((self.n, len(self.x)), dtype=complex)
        for initial in self.initarray:
            x0, v0, a0, g0 = initial
            psi_x = self.hellerPacket(x0, v0, a0, g0)
            sch = Schrodinger(self.x, psi_x, self.v, hbar=self.hbar, m=self.m)
            for i in range(self.n):
                if i != 0:
                    sch.evolve_t(N_step, self.dt)
                psi_sol = sch.psi_x
                psi_comb[i] += psi_sol / np.sqrt(self.N_packet)
        self.psi_comb = psi_comb
        return psi_comb

    def onePacket(self, vkick, init):
        nhalf = int(self.n / 2)
        psi_arr = np.zeros((self.n, len(self.x)), dtype=complex)

        x0, v0, a0, g0 = init
        psi_init = self.hellerPacket(x0, v0, a0, g0)
        tempsch = Schrodinger(self.x, psi_init, self.v, hbar=self.hbar, m=self.m)
        for i in range(nhalf):
            tempsch.evolve_t(1, dt=self.dt)
            psi_arr[i] = tempsch.psi_x

        tempsch.momentum_kick(self.m * vkick / self.hbar)
        for i in range(nhalf):
            tempsch.evolve_t(1, dt=self.dt)
            psi_arr[i + nhalf] = tempsch.psi_x
        return psi_arr

    def interferometry(self, kick, psii=False):
        init1 = self.initarray[0]
        init2 = self.initarray[1]

        if psii:
            psi_1 = self.hellerPacket(init1[0], init1[1], init1[2], init1[3])
            psi_2 = self.hellerPacket(init2[0], init2[1], init2[2], init2[3])
            psi_init = 1 / np.sqrt(self.N_packet) * (psi_1 + psi_2)

        init1[1] = kick
        init2[1] = -kick

        psi1 = self.onePacket(-2 * kick, init1)
        psi2 = self.onePacket(2 * kick, init2)

        psi_comb = (psi1 + psi2) / np.sqrt(self.N_packet)

        if psii:
            psi_comb = np.insert(psi_comb, 0, psi_init, axis=0)

        self.psi_comb = psi_comb
        return self.psi_comb

    def modSquare(self):
        shape = np.shape(self.psi_comb)
        self.psi_squared = np.zeros(shape)

        for i, v in enumerate(self.psi_comb):
            self.psi_squared[i] = np.real(np.conjugate(v) * v)
        return self.psi_squared


class Comparison():

    def __init__(self, n, dt, x, s_psi, h_psi):
        self.x = x
        self.n = n
        self.dt = dt
        self.tl = np.asarray([i * dt for i in range(n)])

        self.s_psi = s_psi
        self.h_psi = h_psi

        self.s_psi_sq = self.modSquare(s_psi)
        self.h_psi_sq = self.modSquare(h_psi)

    def plotInit(self, diff=False, extra=[]):
        print(extra)
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
        axs[0].plot(self.x, self.s_psi_sq[0])
        axs[0].plot(self.x, self.h_psi_sq[0])
        axs[1].plot(self.x, self.s_psi_sq[-1])
        axs[1].plot(self.x, self.h_psi_sq[-1])

        if extra != []:
            axs[0].plot(self.x, extra[0])
            axs[1].plot(self.x, extra[-1])

        axs[0].set_title("Initial")
        axs[1].set_title("Final")
        plt.show()

        if diff:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
            axs[0].plot(self.x, self.s_psi_sq[0] - self.h_psi_sq[0])
            axs[1].plot(self.x, self.s_psi_sq[-1] - self.h_psi_sq[-1])

            axs[0].set_title("Initial Difference")
            axs[1].set_title("Final Difference")
            plt.show()

    def plotHalf(self):
        plt.title("Half")
        plt.plot(self.x, self.s_psi_sq[int(self.n / 2)])
        plt.plot(self.x, self.h_psi_sq[int(self.n / 2)])
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x)|^2$")
        plt.show()

    def modSquare(self, psi_arr):
        shape = np.shape(psi_arr)
        squared = np.zeros(shape)

        for i, v in enumerate(psi_arr):
            squared[i] = np.real(np.conjugate(v) * v)
        return squared

    def overlapArr(self, arr1, arr2):
        tempoverlap = np.zeros(self.n)
        for i in range(self.n):
            psi_comb = np.real(np.conjugate(arr1[i]) * arr2[i])
            tempoverlap[i] = simps(psi_comb, self.x)
        return tempoverlap

    def helNorm(self):
        heloverlap = self.overlapArr(self.h_psi, self.h_psi)
        plt.plot(self.tl, heloverlap)
        plt.show()
        return 0

    def schNorm(self):
        schoverlap = self.overlapArr(self.s_psi, self.s_psi)
        plt.plot(self.tl, schoverlap)
        plt.show()
        return 0

    def bothNorm(self):
        h_overlap = self.overlapArr(self.h_psi, self.h_psi)
        s_overlap = self.overlapArr(self.s_psi, self.s_psi)
        plt.plot(self.tl, h_overlap, label="Heller")
        plt.plot(self.tl, s_overlap, label="Schrodinger")
        plt.legend()
        plt.show()

    def fullOverlap(self):
        fulloverlap = self.overlapArr(self.h_psi, self.s_psi)
        plt.plot(self.tl, fulloverlap)
        plt.show()
        return 0

    def compTo(self, arr):
        heloverlap = self.overlapArr(self.h_psi, arr)
        schoverlap = self.overlapArr(self.s_psi, arr)
        plt.plot(self.tl, heloverlap, label="Heller")
        plt.plot(self.tl, schoverlap, label="Schrodinger")
        plt.legend()
        plt.show()

    def animate(self, potential=[]):
        maximum = max(max(self.h_psi_sq[0]), max(self.s_psi_sq[0]))
        fig, ax = plt.subplots()
        helline, = ax.plot(self.x, self.h_psi_sq[0], label="Heller")
        schroline, = ax.plot(self.x, self.s_psi_sq[0], label="Schrodinger")
        time_text = ax.text(self.x[int(0.01 * self.n)], maximum * 0.9, 't', fontsize=13)
        if potential != []:
            ax.plot(self.x, potential)
        ax.set_xlim((self.x[0], self.x[-1]))
        ax.set_ylim((0, maximum))
        ax.legend(loc=1)

        def initFunc():
            helline.set_data(self.x, self.h_psi_sq[0])
            schroline.set_data(self.x, self.s_psi_sq[0])
            time_text.set_text(f"t=0")
            return helline, schroline, time_text,

        def update(i):
            # if i >= Ntot:
            #     i = 0
            helline.set_data(self.x, self.h_psi_sq[i % self.n])
            schroline.set_data(self.x, self.s_psi_sq[i % self.n])
            time_text.set_text(f"t={self.tl[i % self.n]:.2f}")
            # line.set_data(x, np.sin(x + i / 100))

            return helline, schroline, time_text

        animation.FuncAnimation(fig, update, init_func=initFunc, interval=1, blit=True, save_count=50)
        plt.show()

        return 0


if __name__ == "__main__":
    def derivs(t, current, args, eta, dt):
        m, hbar = args
        x = current[1]
        v = 0
        a = (-2 * current[2] ** 2 / m)
        g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2
        return x, v, a, g


    N = 2 ** 10
    dx = 0.1
    x_length = N * dx
    x = np.linspace(0, x_length, N)
    x = np.zeros(N)
    for i in range(0, N):
        x[i] = i * dx
    x0 = int(3 / 8 * x_length)

    x01 = int(5 / 8 * x_length)

    m = 1
    hbar = 1
    args = (m, hbar)

    d = 4
    k_initial = 2

    p0 = k_initial * hbar
    p0 = 0
    a0 = 1j * hbar / (4 * d ** 2)
    g0 = (1j * hbar / 4) * np.log(2 * np.pi * d ** 2)

    t = 0
    dt = 0.01
    step = 1
    Ns = 2000
    Ntot = int(Ns * step)
    N_half = int(Ntot / 2)

    ####### Standing packets spreadding
    # init = [x0, p0 / m, a0, g0]
    # init2 = [x01, p0 / m, a0, g0]
    # init_array = np.array([init, init2])
    #
    # tupac = AdditionalPacket(Ntot, dt, init_array, derivs, args, x)
    # tupac.runAll()
    # mods = tupac.modSquare()
    # print(simps(mods[0], x))
    # print(simps(mods[-1], x))
    # plt.title("Standing Wave Packet Interference (Heller)")
    # plt.plot(x, mods[0], label="t=0")
    # # plt.plot(x, mods[N_half], label=f"t={int(dt*N_half)}")
    # plt.plot(x, mods[-1], label=f"t={int(dt*Ntot)}")
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel(r"$|\psi(x)|^2$")
    # plt.xlim(min(x), max(x))
    # plt.ylim(0, max(mods[0]))
    # plt.savefig("Heller_intef.png")
    # plt.show()

    ####### Full interference
    # pkick = k_initial * hbar
    # p0 = pkick
    # p0 = 0
    # x0 = x01 = int(1 / 2 * x_length)
    # init = [x0, p0 / m, a0, g0]
    # init2 = [x01, -p0 / m, a0, g0]
    # init_array = np.array([init, init2])
    #
    # tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)

    # psi_1 = tupac.hellerPacket(init[0], init[1], init[2], init[3])
    # psi_2 = tupac.hellerPacket(init2[0], init2[1], init2[2], init2[3])
    # psi_comb = 1 / np.sqrt(2) * (psi_1 + psi_2)
    # plt.plot(x, psi_comb * np.conjugate(psi_comb))
    # plt.show()

    ####### Heller Plot with two packets
    # psi_arr1, tl1, xl1, vl1, al1, gl1 = tupac.onePacket(-2 * pkick / m, init, plot=True, dat=True)
    # psi_arr2, tl2, xl2, vl2, al2, gl2 = tupac.onePacket(2 * pkick / m, init2, plot=True, dat=True)
    # fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
    # fig.suptitle(f"Heller Multiple Packet Simulation")
    # axs[0, 0].plot(tl1, xl1)
    # axs[0, 1].plot(tl1, vl1)
    # axs[1, 0].plot(tl1, np.imag(al1))
    # axs[1, 1].plot(tl1, gl1)
    # axs[0, 0].plot(tl2, xl2,linestyle="--")
    # axs[0, 1].plot(tl2, vl2,linestyle="--")
    # axs[1, 0].plot(tl2, np.imag(al2),linestyle="--")
    # axs[1, 1].plot(tl2, gl2,linestyle="--")
    # axs[0, 0].set_title("X Pos")
    # axs[0, 1].set_title("V Pos")
    # axs[1, 0].set_title("Alpha (Imaginary Part)")
    # axs[1, 1].set_title("Gamma")
    # plt.savefig("Heller_split_intef.png")
    # plt.show()

    # plt.plot(x, psi_arr1[0] * np.conjugate(psi_arr1[0]))
    # plt.plot(x, psi_arr1[-1] * np.conjugate(psi_arr1[-1]))
    # plt.show()

    # tupac.interferometry()
    # mods = tupac.modSquare()
    # plt.plot(x, mods[0])
    # plt.plot(x, mods[-1])
    # plt.show()

    # tupac.interferometry2(pkick, psii=True)
    # mods = tupac.modSquare()
    # plt.title("Wave packet interference for time T")
    # plt.plot(x, mods[0], label="t=0")
    # plt.plot(x, mods[N_half], label="t=T/2")
    # plt.plot(x, mods[-1], label="t=T")
    # plt.xlabel("x")
    # plt.ylabel(r"$|\psi(x)|^2$")
    # plt.xlim(min(x), max(x))
    # plt.ylim(0, max(mods[0]))
    # plt.legend()
    # plt.savefig("Heller_split_wave.png")
    # plt.show()

    ###################################################### Schrodinger

    ##### Standing packet
    # d = 2
    # x0 = int(3 / 8 * x_length)
    #
    # x01 = int(5 / 8 * x_length)
    #
    # init = [x0, p0 / m, a0, g0]
    # init2 = [x01, p0 / m, a0, g0]
    # init_array = np.array([init, init2])
    #
    # V = np.zeros(len(x))
    #
    # sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
    # sch_tupac.runAll()
    #
    # hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
    # hel_tupac.runAll()
    #
    # comp = Comparison(Ntot, dt, x, sch_tupac.psi_comb, hel_tupac.psi_comb)
    # comp.plotInit()
    # comp.bothNorm()
    # comp.fullOverlap()

    ##### Packet in motion
    # x0 = int(3 / 8 * x_length)
    # p0 = k_initial * hbar / 2
    #
    # init = [x0, p0 / m, a0, g0]
    # init_array = np.array([init])
    # V = np.zeros(len(x))
    #
    # sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
    # sch_tupac.runAll(N_step=step)
    #
    # hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
    # hel_tupac.runAll()
    #
    # comp = Comparison(Ntot, dt, x, sch_tupac.psi_comb, hel_tupac.psi_comb)
    # comp.plotInit()
    # comp.bothNorm()
    # comp.fullOverlap()

    ########### Full interference
    d = 2
    p0 = 0
    x0 = x01 = int(1 / 2 * x_length)
    p0 = 0
    init = [x0, p0 / m, a0, g0]
    init2 = [x01, p0 / m, a0, g0]
    init_array = np.array([init, init2])

    V = np.zeros(len(x))
    vkick = k_initial * hbar / m

    ######## One Packet
    # init_temp = [x0, vkick, a0, g0]
    # sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
    # psi_arr = sch_tupac.onePacket(-2 * vkick, init_temp)
    # sch_tupac.psi_comb = psi_arr
    #
    # init_temp = [x0, vkick, a0, g0]
    # hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
    # hel_arr = hel_tupac.onePacket(-2 * vkick, init_temp)
    # hel_tupac.psi_comb = hel_arr
    #
    # comp = Comparison(Ntot, dt, x, sch_tupac.psi_comb, hel_tupac.psi_comb)
    # comp.plotInit()
    # comp.bothNorm()
    # comp.fullOverlap()

    ###### Two packet
    # sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
    # sch_tupac.interferometry(vkick,psii=True)
    # mods = sch_tupac.modSquare()
    # plt.plot(x, mods[0], label="t=0")
    # plt.plot(x, mods[N_half], label="t=T/2")
    # plt.plot(x, mods[-1], label="t=T")
    # plt.legend()
    # plt.show()

    ###### Full interference comp
    sch_tupac = SchroInterference(Ntot, dt, x, V, init_array, hbar=hbar, m=m)
    sch_tupac.interferometry(vkick)
    sch_mods = sch_tupac.modSquare()

    hel_tupac = HellerInterference(Ntot, dt, init_array, derivs, args, x)
    hel_tupac.interferometry2(vkick)
    hel_mods = hel_tupac.modSquare()

    # comp = Comparison(Ntot, dt, x, sch_tupac.psi_comb, hel_tupac.psi_comb)
    # comp.plotInit()
    # comp.plotHalf()
    # comp.fullOverlap()
    # comp.animate()
