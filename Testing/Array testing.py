import numpy as np
import matplotlib.pyplot as plt


def add(x, y):
    return x + y


x = np.arange(1, 5, 1)
y = np.arange(1, 9, 2)

print(x, y)

X, Y = np.meshgrid(x, y)
print(X)
print(Y)

print(add(X, Y)[::-1])

# fig, ax = plt.subplots()
# ax.imshow(add(X, Y), extent=[x.min()-0.5, x.max()+0.5, y.min(), y.max()])
# plt.show()
