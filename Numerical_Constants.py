class Constants():
    def __init__(self, A, dt, dx, k_init):
        self.A = A
        self.dt = dt
        self.dx = dx
        self.k = k_init

        self.get_diff()
        self.get_vdt()
        self.get_kdx()

        self.print_stats()

    def get_diff(self):
        self.diff = self.dt / (self.dx ** 2)

    def get_vdt(self):
        self.vdt = self.A * self.dt

    def get_kdx(self):
        self.kdx = self.k * self.dx

    def print_stats(self):
        print("Diffusion : ", self.diff)
        print("vdt : ", self.vdt)
        print("kdx : ", self.kdx)
