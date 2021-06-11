import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft, ifft, fftfreq, fftshift

plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\FFmpeg\\bin\\ffmpeg.exe'


class Animate():
    """
    Class that carries out the animating of the wave function evolution
    """

    def __init__(self, sch, V, step, dt, lim1=None, lim2=None, title=None, frames=None, scale=(1, 1, 1)):
        """
        Initialising the Wave function
        :param sch: Schrodinger class that the Anminate class will evolve
        :param V: Potential
        :param step: Number of steps of evolution to carry out before before
        :param dt: Time step to do each evolution with
        :param lim1: Tuple of limits for x space plot
        :param lim2: Tuple of limits for k space plot
        :param title: USE TO SAVE ANIMATION. Title of plot to be used to save the animation.
        """
        self.sch = sch
        self.V = V
        self.title = title

        self.step = step
        self.dt = dt
        if frames != None:
            self.frames = frames
        else:
            self.frames = 380

        self.fig = plt.figure()
        self.fig.subplots_adjust(hspace=0.4)
        self.ax1 = self.fig.add_subplot(211)
        self.line1, = self.ax1.plot([], [], label=r'$|\psi(x)|^2$')

        self.potential, = self.ax1.plot([], [], label=r'$V(x)$')
        self.time_text = self.ax1.text(.7, .9, '', fontsize=10)
        self.ax1.legend(loc=1)
        self.ax1.set_xlabel('x')
        self.ax1.set_title(title)

        self.ax2 = self.fig.add_subplot(212)
        self.line2, = self.ax2.plot([], [], label=r'$|\psi(k)|$')
        self.ax2.legend()
        self.ax2.set_xlabel('k')

        if (lim1 != None):
            self.time_text = self.ax1.text(.7, lim1[1][1] - .1, '', fontsize=10)
            self.ax1.set_xlim(lim1[0])
            self.ax1.set_ylim(lim1[1])

        else:
            lim1 = ((0, self.sch.x_length), (0, max(np.real(self.sch.psi_squared))))
            self.time_text = self.ax1.text(.7, lim1[1][1] - .1, '', fontsize=10)
            self.ax1.set_xlim(lim1[0])
            self.ax1.set_ylim(lim1[1])

        if (lim2 != None):
            self.ax2.set_xlim(lim2[0])
            self.ax2.set_ylim(lim2[1])

        else:
            lim2 = ((min(self.sch.k), max(self.sch.k)), (0, 1))
            self.ax2.set_xlim(lim2[0])
            self.ax2.set_ylim(lim2[1])

    def init(self):
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.potential.set_data([], [])
        return self.line1, self.line2, self.potential,

    def animate(self, i, schro=None, func=None, args=None):
        self.sch.evolve_t(self.step, self.dt)

        self.line1.set_data(self.sch.x, self.sch.mod_square_x(True))
        self.line1.set_label('This')
        self.time_text.set_text(f"t = {self.sch.t:.3f}")
        # self.potential.set_data(self.sch.x, self.sch.v / max(self.sch.v))
        self.potential.set_data(self.sch.x, self.sch.v)
        self.line2.set_data(fftshift(self.sch.k), (fftshift(self.sch.mod_square_k(r=True))))

        if schro != None:
            schro.evolve_t(self.step, self.dt)
            self.line3.set_data(schro.x, schro.mod_square_x(True))

        elif func != None:
            xt, pt, at, gt, psi_h = func(self.sch, args)
            mod_psi_h = psi_h * np.conjugate(psi_h)
            self.line3.set_data(self.sch.x, mod_psi_h)

        else:
            self.line3.set_data([],[])

        return self.line1, self.line2, self.line3, self.potential, self.time_text,

    def make_fig(self, schro=None, func=None, args=None, label=""):
        self.init()

        if schro != None or func != None:

            if label != "":
                self.line3, = self.ax1.plot([], [], label=label, linestyle=":")
                self.ax1.legend(loc=1)

        else:
            self.line3, = self.ax1.plot([], [], linestyle=":")

        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames=self.frames, interval=30, blit=True, fargs=(schro, func, args))

        if self.title != None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(self.title + '.mp4', writer=writer)
        plt.show()


if __name__ == "__main__":
    a = Animate(lim1=((0, 10), (-1, 1)), lim2=((0, 10), (-1, 1)))

    a.make_fig()
