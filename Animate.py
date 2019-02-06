import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft, ifft, fftfreq, fftshift
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\FFmpeg\\bin\\ffmpeg.exe'

class Animate():

    def __init__(self, sch, V, step, dt, lim1=None, lim2=None, title=None):
        self.sch = sch
        self.V = V
        self.title = title

        self.step = step
        self.dt = dt
        self.frames = 170

        self.fig = plt.figure()
        self.fig.subplots_adjust(hspace=0.4)
        self.ax1 = self.fig.add_subplot(211)
        self.line1, = self.ax1.plot([], [], label=r'$|\psi(x)|^2$')
        self.potential, = self.ax1.plot([], [], label=r'$V(x)$')
        self.time_text = self.ax1.text(.7, lim1[1][1]-.1, '', fontsize=10)
        self.ax1.legend()
        self.ax1.set_xlabel('x')
        self.ax1.set_title(title)

        self.ax2 = self.fig.add_subplot(212)
        self.line2, = self.ax2.plot([], [], label=r'$|\psi(k)|$')
        self.ax2.legend()
        self.ax2.set_xlabel('k')

        if (lim1 != None):
            self.ax1.set_xlim(lim1[0])
            self.ax1.set_ylim(lim1[1])

        if (lim2 != None):
            self.ax2.set_xlim(lim2[0])
            self.ax2.set_ylim(lim2[1])

    def init(self):
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.potential.set_data([], [])
        return self.line1, self.line2, self.potential,

    def animate(self, i):
        self.sch.evolve_t(self.step, self.dt)

        # print(self.sch.t)
        # print(self.sch.norm_x())

        self.line1.set_data(self.sch.x, self.sch.mod_square_x(True))
        self.line1.set_label('This')
        self.time_text.set_text(f"t = {self.sch.t:.3f}")
        self.potential.set_data(self.sch.x, self.V)
        self.line2.set_data(fftshift(self.sch.k), fftshift(abs(self.sch.psi_k)))
        return self.line1, self.line2, self.potential, self.time_text,

    def make_fig(self):
        self.init()

        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames=self.frames, interval=30, blit=True)

        if self.title!=None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(self.title +'.mp4', writer=writer)
        plt.show()


if __name__ == "__main__":
    a = Animate(lim1=((0, 10), (-1, 1)), lim2=((0, 10), (-1, 1)))

    a.make_fig()
