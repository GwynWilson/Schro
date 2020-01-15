import Ito_Process as ito
import matplotlib.pyplot as plt
import numpy as np


def a(x, t):
    return 0


def b(x, t):
    return amp

I = 0
amp = 10 ** (-5)

time = 10
dt = 10**-3
n = int(time / dt)

# run = ito.itoProcess(n, dt, a, b, x0=I)
# plt.plot(run)
# plt.show()


Bmag = 10
thet = np.pi/4
print(np.cos(thet))

B1 = np.zeros(2)
# run1 = ito.itoProcess(n, dt, a, b, x0=I)
run1 = 1
B1[0] = Bmag*run1
print(B1)

B2 = np.zeros(2)
# run2 = ito.itoProcess(n, dt, a, b, x0=I)
run2 = 1
B2[0] = Bmag*run2*np.cos(thet)
B2[1] = Bmag*run2*np.sin(thet)

x1,y1 = B1
plt.scatter(x1,y1)

x2,y2 = B2
plt.scatter(x2,y2)

plt.scatter(x1+x2,y1+y2)
plt.show()