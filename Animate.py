import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Animate():

    def __init__(self,sch, V, step, dt, lim1=None, lim2=None):
        self.sch = sch
        self.V = V

        self.step = step
        self.dt = dt
        self.frames = 120

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.line1, = self.ax1.plot([], [])
        self.potential, = self.ax1.plot([], [])

        self.ax2 = self.fig.add_subplot(212)
        self.line2, = self.ax2.plot([], [])

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
        self.line1.set_data(self.sch.x, self.sch.mod_square_x(True))
        self.potential.set_data(self.sch.x, self.V)
        self.line2.set_data(self.sch.k, abs(self.sch.psi_k))
        return self.line1, self.line2, self.potential,

    def make_fig(self):
        self.init()

        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames=self.frames, interval=30, blit=True)

        plt.show()


if __name__ == "__main__":
    a = Animate(lim1=((0, 10), (-1, 1)), lim2=((0, 10), (-1, 1)))

    a.make_fig()
