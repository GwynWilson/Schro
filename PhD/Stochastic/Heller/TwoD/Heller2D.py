import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\FFmpeg\\bin\\ffmpeg.exe'


class Heller2D:
    def __init__(self, n, dt, init, func):
        self.n = n
        self.dt = dt
        self.init = init

        self.derivatives = func
        self.args = None
        self.noise = None

        self.variables_arr = np.zeros((len(init), n), dtype=complex)
        self.tl = np.zeros(n)
        return None

    def genNoise(self):
        return np.random.randn(self.n) / np.sqrt(self.dt)

    def rk4(self, args, noise=None, set=False):
        self.noise = noise
        self.args = args
        temp_var_arr = np.zeros((len(self.init), self.n), dtype=complex)
        temp_var_arr[:, 0] = self.init
        # print(temp_var_arr)
        current = np.asarray(self.init, dtype=complex)

        t = 0
        tl = np.zeros(self.n)

        if noise is None:
            self.noise = self.genNoise()

        for i in range(self.n - 1):
            N = self.noise[i]
            # eta = np.random.randn()
            k0 = np.asarray(self.derivatives(t, current, self.args, N, self.dt))
            k1 = np.asarray(self.derivatives(t + self.dt / 2, current + k0 / 2, self.args, N, self.dt))
            k2 = np.asarray(self.derivatives(t + self.dt / 2, current + k1 / 2, self.args, N, self.dt))
            k3 = np.asarray(self.derivatives(t + self.dt, current + k2, args, N, self.dt))

            t += self.dt
            for j in range(len(self.init)):
                current[j] += (k0[j] + 2 * k1[j] + 2 * k2[j] + k3[j]) / 6
            temp_var_arr[:, i + 1] = current
            tl[i + 1] = t
        if set:
            self.variables_arr = temp_var_arr
            self.tl = tl
        return tl, temp_var_arr


class PlotTools2D():
    def __init__(self, tl, var, xlim, ylim, n_point):
        self.tl = tl
        self.var = var
        self.xlim = xlim
        self.ylim = ylim
        self.n_point = n_point

        self.n = np.shape(var)[1]  # maybe

        self.x = np.linspace(-xlim, xlim, n_point)
        self.y = np.linspace(ylim, -ylim, n_point) # The y axis is inverted in the plots
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def hellerGaussian2D(self, x, y, vals, args):
        # m, hbar, w1, w2 = args
        xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
        return np.exp((1j / args[1]) * (
                axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + args[0] * vxt * (x - xt) + args[0] * vyt * (
                y - yt) + gt))

    def plot2D(self, vals, args):
        Z = self.hellerGaussian2D(self.X, self.Y, vals, args)
        psi_s = np.real(Z * np.conjugate(Z))
        fig, ax = plt.subplots()
        data = ax.imshow(psi_s, extent=[-self.xlim, self.xlim, -self.ylim, self.ylim])
        cb = fig.colorbar(data)
        plt.show()

    def animation2D(self, args,save=""):
        Z = self.hellerGaussian2D(self.X, self.Y, self.var[:, 0], args)
        psi_s = np.real(Z * np.conjugate(Z))

        fig, ax = plt.subplots()
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        # data = ax.pcolormesh(x, y, np.asarray(psi_s))
        # data = ax.imshow(psi_s, extent=[-self.xlim, self.xlim, -self.ylim, self.ylim], cmap="gist_heat")
        data = ax.imshow(psi_s, extent=[-self.xlim, self.xlim, -self.ylim, self.ylim])
        time_text = ax.text(-0.95*self.xlim, 0.9*self.ylim, '', fontsize=10,color="w")
        cb = fig.colorbar(data)

        def init():
            data.set_array(np.array(psi_s))
            time_text.set_text("")
            return data

        def animate(i):
            index = i % self.n
            # print(index)
            time_text.set_text(f"t = {self.tl[index]:.3f}")

            Z = self.hellerGaussian2D(self.X, self.Y, self.var[:, index], args)
            psi_s = np.real(Z * np.conjugate(Z))

            data.set_array(np.array(psi_s))
            return data

        ani = animation.FuncAnimation(fig, animate, interval=100, frames=int(self.n), blit=False, init_func=init)

        if save!="":
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'{save}.mp4', writer=writer)

        plt.show()


if __name__ == "__main__":
    def hellerGaussian2D(x, y, vals, args):
        m, hbar, w1, w2 = args
        xt, yt, vxt, vyt, axt, ayt, lamt, gt = vals
        return np.exp((1j / hbar) * (
                axt * (x - xt) ** 2 + ayt * (y - yt) ** 2 + lamt * (x - xt) * (y - yt) + m * vxt * (x - xt) + m * vyt * (
                y - yt) + gt))


    def derivsdt(t, current, args, eta, dt):
        m, hbar, w1, w2 = args
        x = current[2] * dt
        y = current[3] * dt
        vx = -m * w1 ** 2 * current[0]*dt
        vy = -m * w2 ** 2 * current[1]*dt
        ax = (-(2 / m) * current[4] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w1 ** 2)*dt
        ay = (-(2 / m) * current[5] ** 2 - (1 / (2 * m)) * current[6] ** 2 - 0.5 * m * w2 ** 2)*dt
        lam = (-2 * ((current[4] / m) + (current[5] / m)) * current[6])*dt
        gam = (1j * hbar * current[4] / m + 1j * hbar * current[5] / m + 0.5 * m * current[2] ** 2 + 0.5 * m * current[
            3] ** 2 - 0.5 * m * w1 ** 2 * current[0] ** 2 - 0.5 * m * w2 ** 2 * current[1] ** 2)*dt
        return x, y, vx, vy, ax, ay, lam, gam


    n = 100
    dt = 0.001

    w = 10
    m = 1
    hbar = 1
    args = (m, hbar, w, w)
    temp = 0.5 * m * w ** 2

    x0 = 0.5
    y0 = 0.25

    vx0 = 0
    vy0 = 0

    lam = 0

    sigx = np.sqrt(hbar / (2 * m * w))
    sigy = np.sqrt(hbar / (2 * m * w))

    ax0 = 1j * hbar / (4 * sigx ** 2)
    ay0 = 1j * hbar / (4 * sigy ** 2)

    sigt = np.sqrt(sigx ** 2 + sigy ** 2)

    g0 = (1j * hbar / 2) * np.log(2 * np.pi * sigx * sigy)

    init = [x0, y0, vx0, vy0, ax0, ay0, lam, g0]

    #############################
    Hel = Heller2D(n, dt, init, derivsdt)
    tl, var_arr = Hel.rk4(args, set=True)

    # plt.plot(Hel.tl,Hel.variables_arr[-1])
    # plt.show()

    ###########################
    xlim = 2
    ylim = 2
    n_point = 100

    plot = PlotTools2D(tl, var_arr, xlim, ylim, n_point)
    # plot.plot2D(init, args)
    plot.animation2D(args)
