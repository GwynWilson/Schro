import numpy as np
# from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
from scipy.integrate import simps, trapz
import matplotlib.pyplot as plt
from matplotlib import animation


class Schrodinger2D(object):
    def __init__(self, x, y, psi, v, args, t=0):
        self.x = x
        self.dx = x[1] - x[0]
        self.y = y
        self.dy = y[1] - y[0]
        self.psi_x = psi
        self.psi_k = fftshift(fftn(psi))

        self.shape = psi.shape

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.call = False
        self.args = args
        if callable(v):
            self.potential = v
            self.V = self.potential(self.X, self.Y, self.args)
            self.call = True
        else:
            self.V = v

        # self.args = args
        self.m = args[0]
        self.hbar = args[1]

        self.t = t
        self.dt = None

        self.klimx = np.pi / self.dx
        self.kx = -self.klimx + (2 * self.klimx / len(self.x)) * np.arange(len(self.x))
        self.klimy = np.pi / self.dy
        self.ky = -self.klimy + (2 * self.klimy / len(self.y)) * np.arange(len(self.y))

        self.KX, self.KY = np.meshgrid(self.kx, self.ky)

    def evolvex(self):
        if self.call == True:
            # Vxy = self.potential(self.X, self.Y, self.args)
            # self.psi_x = self.psi_x * np.exp(1j * (np.asarray(Vxy) * self.dt) / self.hbar)
            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    vij = self.potential(self.x[i], self.y[j], self.args)
                    self.psi_x[i, j] = self.psi_x[i, j] * np.exp(1j * (vij * self.dt) / self.hbar)


        else:
            Vxy = self.V
            self.psi_x = self.psi_x * np.exp(-1j * (np.asarray(Vxy) * self.dt) / self.hbar)

        # fig, ax = plt.subplots()
        # data = ax.imshow(Vxy)
        # cb = fig.colorbar(data)
        # plt.show()

    def evolvek(self):
        # psiks = np.real(self.psi_k*np.conjugate(self.psi_k))
        # fig, ax = plt.subplots()
        # data = ax.imshow(psiks[::-1], extent=[-self.klimx, self.klimx, -self.klimy, self.klimy])  # The ::-1 is to flip the axes
        # cb = fig.colorbar(data)
        # plt.show()

        self.psi_k = self.psi_k * (
            np.exp(-1j * (self.hbar * (np.asarray(self.KX) ** 2 + np.asarray(self.KY) ** 2) * self.dt) / (2 * self.m)))

    def normk(self):
        squared = np.real(self.psi_k * np.conjugate(self.psi_k))
        norm = simps(simps(squared, self.kx), self.ky)
        self.psi_k = self.psi_k / np.sqrt(norm)

    def normx(self):
        squared = np.real(self.psi_x * np.conjugate(self.psi_x))
        norm = simps(simps(squared, self.x), self.y)
        self.psi_x = self.psi_x / np.sqrt(norm)

    def xExpectation(self):
        psix_s = np.real(self.psi_x * np.conjugate(self.psi_x))
        return simps(simps(self.X * psix_s, self.x), self.y)

    def yExpectation(self):
        psix_s = np.real(self.psi_x * np.conjugate(self.psi_x))
        return simps(simps(self.Y * psix_s, self.y), self.x)

    def xsExpectation(self):
        psix_s = np.real(self.psi_x * np.conjugate(self.psi_x))
        return simps(simps(self.X ** 2 * psix_s, self.x), self.y)

    def ysExpectation(self):
        psix_s = np.real(self.psi_x * np.conjugate(self.psi_x))
        return simps(simps(self.Y ** 2 * psix_s, self.y), self.x)

    def evolvet(self, N_steps, dt):
        self.dt = dt
        for i in range(N_steps):
            self.psi_k = fftshift(fftn(self.psi_x))
            # self.normk()
            self.evolvek()
            self.psi_x = ifftn(fftshift(self.psi_k))
            # self.normx()
            self.evolvex()
        self.t += (N_steps * self.dt)

    def expectationVals(self, N_steps, dt):
        self.dt = dt
        t_list = [i * self.dt for i in range(N_steps)]
        x_av = np.zeros(N_steps)
        y_av = np.zeros(N_steps)
        xs_av = np.zeros(N_steps)
        ys_av = np.zeros(N_steps)

        for i in range(N_steps):
            x_expect = self.xExpectation()
            x_av[i] = x_expect
            y_expect = self.yExpectation()
            y_av[i] = y_expect
            xs_expect = self.xsExpectation()
            xs_av[i] = xs_expect
            ys_expect = self.ysExpectation()
            ys_av[i] = ys_expect

            self.psi_k = fftshift(fftn(self.psi_x))
            self.evolvek()
            self.psi_x = ifftn(fftshift(self.psi_k))
            self.evolvex()
        self.t += (N_steps * self.dt)

        return t_list, x_av, y_av, xs_av, ys_av

    def getRing(self, theta, psis, A):
        xring = A * np.cos(theta)
        yring = A * np.sin(theta)
        psi_list = []
        for xpar, ypar in zip(xring, yring):
            indx = np.abs(self.x - xpar).argmin()
            indy = np.abs(self.y - ypar).argmin()
            psi_list.append(psis[indy, indx])
        return psi_list

    def ringExpectation(self, N_steps, dt, rp):
        theta = np.linspace(-np.pi, np.pi, self.shape[0])
        self.dt = dt
        t_list = [i * self.dt for i in range(N_steps)]
        s_list = []
        s_wid = []

        x_wid = []
        y_wid = []

        for i in range(N_steps):
            expect_x = self.xExpectation()
            expect_y = self.yExpectation()
            squared = np.real(self.psi_x * np.conjugate(self.psi_x))
            psi_ring = self.getRing(theta, squared, rp)

            angle1 = np.arctan2(expect_y, expect_x)
            theta1 = theta - angle1
            if abs(angle1) > np.pi / 2:
                first = psi_ring[:int(len(psi_ring) / 2)]
                second = psi_ring[int(len(psi_ring) / 2):]
                psi_ring = np.concatenate((second, first))
                if angle1 > 0:
                    theta1 += np.pi
                else:
                    theta1 -= np.pi

            expect_xwid = np.sqrt(self.xsExpectation() - expect_x ** 2)
            x_wid.append(expect_xwid)
            expect_ywid = np.sqrt(self.ysExpectation() - expect_y ** 2)
            y_wid.append(expect_ywid)

            s = theta1 * rp
            # plt.plot(s,psi_ring)
            # plt.show()

            s_ex = simps(psi_ring * s, s)  # Should always be zero
            s_s_ex = simps(psi_ring * s ** 2, s)
            wids = np.sqrt(s_s_ex - s_ex ** 2)

            s_list.append(s_ex)
            s_wid.append(wids)

            self.psi_k = fftshift(fftn(self.psi_x))
            self.evolvek()
            self.psi_x = ifftn(fftshift(self.psi_k))
            self.evolvex()

        self.t += (N_steps * self.dt)

        # return t_list, thet_list, thet_wid
        return t_list, s_list, s_wid, x_wid, y_wid


class PlotTools():
    def __init__(self):
        return None

    def hellerGaussian2D(self, x, y, vals, args):
        m = args[0]
        hbar = args[1]
        xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
        return np.exp((1j / hbar) * (
                axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (
                x - xt) + m * vyt * (
                        y - yt) + gt))

    def expectationValues(self, t_l, x_l, y_l, xs_l, ys_l, show=True):
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

        ax[0, 0].plot(t_l, x_l)
        ax[0, 1].plot(t_l, y_l)

        ax[1, 0].plot(t_l, np.sqrt(xs_l - x_l ** 2))
        ax[1, 1].plot(t_l, np.sqrt(ys_l - y_l ** 2))

        ax[0, 0].set_ylabel(r"$\langle x(t) \rangle$")
        ax[0, 1].set_ylabel(r"$\langle y(t) \rangle$")
        ax[1, 0].set_ylabel(r"$\Delta x(t)$")
        ax[1, 1].set_ylabel(r"$\Delta y(t)$")

        ax[1, 0].set_xlabel(r"$t$")
        ax[1, 1].set_xlabel(r"$t$")

        if show:
            plt.show()
            return None
        else:
            return fig, ax
        # plt.show()

    def expectationValuesComparrison(self, t_l, x_l, y_l, xs_l, ys_l, x_c, y_c, xw_c, yw_c, show=True):
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

        ax[0, 0].plot(t_l, x_l)
        ax[0, 1].plot(t_l, y_l)

        ax[1, 0].plot(t_l, np.sqrt(xs_l - x_l ** 2))
        ax[1, 1].plot(t_l, np.sqrt(ys_l - y_l ** 2))

        ax[0, 0].plot(t_l, x_c, linestyle="--")
        ax[0, 1].plot(t_l, y_c, linestyle="--")

        ax[1, 0].plot(t_l, xw_c, linestyle="--")
        ax[1, 1].plot(t_l, yw_c, linestyle="--")

        ax[0, 0].set_ylabel(r"$\langle x(t) \rangle$")
        ax[0, 1].set_ylabel(r"$\langle y(t) \rangle$")
        ax[1, 0].set_ylabel(r"$\Delta x(t)$")
        ax[1, 1].set_ylabel(r"$\Delta y(t)$")

        ax[1, 0].set_xlabel(r"$t$")
        ax[1, 1].set_xlabel(r"$t$")

        if show:
            plt.show()
            return None
        else:
            return fig, ax

    def compareInit(self, psi1, psi2, xlen, klen, other=[], save=""):
        psi1k = fftshift(fftn(psi1))
        Z1 = np.real(psi1 * np.conjugate(psi1))
        Z1k = np.real(psi1k * np.conjugate(psi1k))

        psi2k = fftshift(fftn(psi2))
        Z2 = np.real(psi2 * np.conjugate(psi2))
        Z2k = np.real(psi2k * np.conjugate(psi2k))

        fig, ax = plt.subplots(1, 2, figsize=(8, 5))

        data = ax[0].imshow(Z1[::-1], extent=[-xlen, xlen, -xlen, xlen])
        ax[1].imshow(Z2[::-1], extent=[-xlen, xlen, -xlen, xlen])

        ax[0].set_title("Initial")
        ax[1].set_title("Final")
        plt.suptitle("Position space")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if other!=[]:
            otherx, othery = other
            ax[0].plot(otherx,othery, linestyle="--", color="w")
            ax[1].plot(otherx, othery, linestyle="--", color="w")

        if save != "":
            plt.savefig(save + "_xspace.png")
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        data = ax[0].imshow(Z1k[::-1], extent=[-klen, klen, -klen, klen])
        ax[1].imshow(Z2k[::-1], extent=[-klen, klen, -klen, klen])
        ax[0].set_title("Initial")
        ax[1].set_title("Final")
        plt.suptitle("Momentum space")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save != "":
            plt.savefig(save + "_kspace.png")
        plt.show()

    def animation2D(self, sch, nstep, dt, args, xlim, klim, save=""):
        Z = sch.psi_x
        psi_s = np.real(Z * np.conjugate(Z))

        Zk = sch.psi_k
        psik_s = np.real(Zk * np.conjugate(Zk))

        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        # data = ax.pcolormesh(x, y, np.asarray(psi_s))
        # data = ax.imshow(psi_s, extent=[-self.xlim, self.xlim, -self.ylim, self.ylim], cmap="gist_heat")
        data = ax[0].imshow(psi_s[::-1], extent=[-xlim, xlim, -xlim, xlim])
        kdata = ax[1].imshow(psik_s[::-1], extent=[-klim, klim, -klim, klim])
        time_text = ax[0].text(-0.95 * xlim, 0.9 * xlim, '', fontsize=10, color="w")

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(data, cax=cbar_ax)

        # cb = fig.colorbar(data)

        def init():
            data.set_array(np.array(psi_s))
            kdata.set_array(np.asarray(psik_s))
            time_text.set_text("")
            return data

        def animate(i):
            time_text.set_text(f"t = {sch.t:.3f}")

            sch.evolvet(nstep, dt)
            Z = sch.psi_x
            psi_s = np.real(Z * np.conjugate(Z))

            Zk = sch.psi_k
            psik_s = np.real(Zk * np.conjugate(Zk))

            data.set_array(np.array(psi_s)[::-1])
            kdata.set_array(np.asarray(psik_s)[::-1])
            return data

        ani = animation.FuncAnimation(fig, animate, interval=1, blit=False, init_func=init)

        if save != "":
            ani = animation.FuncAnimation(fig, animate, frames=100, interval=1, blit=False, init_func=init)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'{save}.mp4', writer=writer)

        plt.show()

    def hellerAnimation(self, n, X, Y, tl, arr, args, xlim, save=""):
        Z = self.hellerGaussian2D(X, Y, arr[:, 0], args)
        psi_s = np.real(Z * np.conjugate(Z))

        fig, ax = plt.subplots()
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        # data = ax.pcolormesh(x, y, np.asarray(psi_s))
        # data = ax.imshow(psi_s, extent=[-self.xlim, self.xlim, -self.ylim, self.ylim], cmap="gist_heat")
        data = ax.imshow(psi_s[::-1], extent=[-xlim, xlim, -xlim, xlim])
        time_text = ax.text(-0.95 * xlim, 0.9 * xlim, '', fontsize=10, color="w")
        cb = fig.colorbar(data)

        def init():
            data.set_array(np.array(psi_s))
            time_text.set_text("")
            return data

        def animate(i):
            index = i % n
            # print(index)
            time_text.set_text(f"t = {tl[index]:.3f}")

            Z = self.hellerGaussian2D(X, Y, arr[:, index], args)
            psi_s = np.real(Z * np.conjugate(Z))

            data.set_array(np.array(psi_s)[::-1])
            return data

        ani = animation.FuncAnimation(fig, animate, interval=100, frames=int(n), blit=False, init_func=init)

        if save != "":
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'{save}.mp4', writer=writer)

        plt.show()

    def animateComparrison(self, sch, n, tl, arr, args, xlim, klim, save="",rp=0, full=True):
        initial_psi = sch.psi_x
        Zh = self.hellerGaussian2D(sch.X, sch.Y, arr[:, 0], args)
        psih_x = np.real(Zh * np.conjugate(Zh))

        Zhk = fftshift(fftn(Zh))
        psih_k = np.real(Zhk * np.conjugate(Zhk))

        Zs = sch.psi_x
        psis_x = np.real(Zs * np.conjugate(Zs))
        Zsk = fftshift(fftn(Zs))
        psis_k = np.real(Zsk * np.conjugate(Zsk))

        if full:
            fig, ax = plt.subplots(2, 2, figsize=(8, 6))
            heller = ax[0, 0].imshow(psih_x[::-1], extent=[-xlim, xlim, -xlim, xlim])
            schro = ax[0, 1].imshow(psis_x[::-1], extent=[-xlim, xlim, -xlim, xlim])

            heller_k = ax[1, 0].imshow(psih_k[::-1], extent=[-klim, klim, -klim, klim])
            schro_k = ax[1, 1].imshow(psis_k[::-1], extent=[-klim, klim, -klim, klim])

            time_text = ax[0, 0].text(-0.95 * xlim, 0.9 * xlim, '', fontsize=10, color="w")

            # cb = fig.colorbar(heller)

            def init():
                heller.set_array(np.array(psih_x))
                schro.set_array(np.array(psis_x))
                heller_k.set_array(np.array(psih_k))
                schro_k.set_array(np.array(psis_k))

                time_text.set_text("")
                return time_text, heller, schro, heller_k, schro_k

            def animate(i):
                index = i % n
                if index == 0:
                    sch.psi_x = initial_psi

                else:
                    sch.evolvet(1, tl[1] - tl[0])
                time_text.set_text(f"t = {tl[index]:.3f}")

                Zh = self.hellerGaussian2D(sch.X, sch.Y, arr[:, index], args)
                psih_x = np.real(Zh * np.conjugate(Zh))

                Zhk = fftshift(fftn(Zh))
                psih_k = np.real(Zhk * np.conjugate(Zhk))

                Zs = sch.psi_x
                psis_x = np.real(Zs * np.conjugate(Zs))
                Zsk = fftshift(fftn(Zs))
                psis_k = np.real(Zsk * np.conjugate(Zsk))

                heller.set_array(np.array(psih_x)[::-1])
                schro.set_array(np.array(psis_x)[::-1])
                heller_k.set_array(np.array(psih_k)[::-1])
                schro_k.set_array(np.array(psis_k)[::-1])
                return time_text, heller, schro, heller_k, schro_k


        else:
            N = np.shape(psih_x)[0]
            theta = np.linspace(-np.pi, np.pi, N)
            fig, ax = plt.subplots(1, 2, figsize=(8, 5))
            heller = ax[0].imshow(psih_x[::-1], extent=[-xlim, xlim, -xlim, xlim])
            schro = ax[1].imshow(psis_x[::-1], extent=[-xlim, xlim, -xlim, xlim])
            time_text = ax[0].text(-0.95 * xlim, 0.9 * xlim, '', fontsize=10, color="w")
            ring1, = ax[0].plot(rp * np.cos(theta), rp * np.sin(theta), linestyle="--", color="w")
            ring2, = ax[1].plot(rp * np.cos(theta), rp * np.sin(theta), linestyle="--", color="w")
            ax[0].set_title("Heller")
            ax[1].set_title("Split-Step")

            def init():
                heller.set_array(np.array(psih_x))
                schro.set_array(np.array(psis_x))
                time_text.set_text("")
                ring1.set_data(rp * np.cos(theta), rp * np.sin(theta))
                ring2.set_data(rp * np.cos(theta), rp * np.sin(theta))
                return time_text, heller, schro, ring1,ring2

            def animate(i):
                index = i % n
                if index == 0:
                    sch.psi_x = initial_psi
                else:
                    sch.evolvet(1, tl[1] - tl[0])
                time_text.set_text(f"t = {tl[index]:.3f}")

                Zh = self.hellerGaussian2D(sch.X, sch.Y, arr[:, index], args)
                psih_x = np.real(Zh * np.conjugate(Zh))

                Zs = sch.psi_x
                psis_x = np.real(Zs * np.conjugate(Zs))

                heller.set_array(np.array(psih_x)[::-1])
                schro.set_array(np.array(psis_x)[::-1])
                ring1.set_data(rp * np.cos(theta), rp * np.sin(theta))
                ring2.set_data(rp * np.cos(theta), rp * np.sin(theta))
                return time_text, heller, schro,ring1,ring2

        ani = animation.FuncAnimation(fig, animate, interval=100, frames=int(n), blit=False, init_func=init)

        if save != "":
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'{save}.mp4', writer=writer)

        plt.tight_layout()
        plt.show()

    def animateRing(self, sch, nstep, dt, args, xlim, klim, rp, save=""):
        N = sch.shape[0]
        theta = np.linspace(-np.pi, np.pi, N)
        psi_s = np.real(sch.psi_x * np.conjugate(sch.psi_x))

        psi_ring = sch.getRing(theta, psi_s, rp)

        # fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        # data = ax[0].imshow(psi_s[::-1], extent=[-xlim, xlim, -xlim, xlim])
        # ringwav, = ax[1].plot(theta, psi_ring)
        # time_text = ax[0].text(-0.95 * xlim, 0.9 * xlim, '', fontsize=10, color="w")
        #
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(data, cax=cbar_ax)
        #
        # def init():
        #     data.set_array(np.array(psi_s))
        #     ringwav.set_data(theta, psi_ring)
        #     time_text.set_text("")
        #     return data, ringwav, time_text
        #
        # def animate(i):
        #     time_text.set_text(f"t = {sch.t:.3f}")
        #
        #     sch.evolvet(nstep, dt)
        #     Z = sch.psi_x
        #     psi_s = np.real(Z * np.conjugate(Z))
        #
        #     data.set_array(np.array(psi_s)[::-1])
        #     ringwav.set_data(theta, sch.getRing(theta, psi_s, rp))
        #     return data, ringwav, time_text

        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        line1, = ax[1].plot(theta * rp, psi_ring)
        wave = ax[0].imshow(psi_s[::-1], extent=[-xlim, xlim, -xlim, xlim])
        circle, = ax[0].plot(rp * np.cos(theta), rp * np.sin(theta), linestyle="--", color="w")

        def init():
            line1.set_data([], [])
            return line1,

        def animate(i):
            sch.evolvet(nstep, dt)
            Z = sch.psi_x
            psi_s = np.real(Z * np.conjugate(Z))
            wave.set_array(np.asarray(psi_s)[::-1])

            psi_ring = sch.getRing(theta, psi_s, rp)

            expecx = sch.xExpectation()
            expecy = sch.yExpectation()
            angle1 = np.arctan2(expecy, expecx)
            theta1 = theta - angle1
            if abs(angle1) > np.pi / 2:
                # psi_ring= fftshift(psi_ring)
                # theta1 = fftshift(theta1)
                first = psi_ring[:int(N / 2)]
                second = psi_ring[int(N / 2):]
                psi_ring = np.concatenate((second, first))
                if angle1 > 0:
                    theta1 += np.pi
                else:
                    theta1 -= np.pi

            line1.set_data(theta1 * rp, psi_ring)

            circle.set_data(rp * np.cos(theta), rp * np.sin(theta))

        plt.tight_layout()
        ani = animation.FuncAnimation(fig, animate, interval=1, blit=False)

        if save != "":
            ani = animation.FuncAnimation(fig, animate, frames=225, interval=1, blit=False, init_func=init)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'{save}.mp4', writer=writer)

        # plt.tight_layout()
        plt.show()
