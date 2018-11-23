import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Animate():

    def __init__(self, xlim1=None, ylim1=None, xlim2=None, ylim2=None):
        self.x = np.linspace(0, 10, 100)
        self.y = 2 * self.x

        self.frames = 120

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.line1, = self.ax1.plot([], [])

        self.ax2 = self.fig.add_subplot(212)
        self.line2, = self.ax2.plot([], [])

        if (xlim1 != None):
            self.ax1.set_xlim(xlim1)

        if (ylim1 != None):
            self.ax1.set_ylim(ylim1)

        if (xlim2 != None):
            self.ax2.set_xlim(xlim2)

        if (ylim2 != None):
            self.ax2.set_ylim(ylim2)

    def sin(self, i):
        self.y = np.sin(self.x + i)

    def init(self):
        self.line1.set_data([], [])
        self.line2.set_data([],[])
        return self.line1, self.line2,

    def animate(self, i):
        self.sin(i)
        self.line1.set_data(self.x, self.y)
        self.line2.set_data(self.x, -0.5*self.y)
        return self.line1, self.line2,

    def make_fig(self):
        self.init()

        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames=self.frames, interval=30, blit=True)

        plt.show()


a = Animate(xlim1=(0, 10), ylim1=(-1, 1), xlim2=(0, 10), ylim2=(-1, 1))

a.make_fig()
