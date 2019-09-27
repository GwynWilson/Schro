import numpy as np
import matplotlib.pyplot as plt
import glob

alltxt = glob.glob("*.txt")
GBox = []
exclude = ["3", "5", "7", "9"]

for i in alltxt:
    if i.startswith(str("GBox_")):
        GBox.append(i)

for i in GBox:
    name = list(i.split("_"))[1][:-4]
    if name not in exclude:
        dat = np.loadtxt(i)
        t_list, x_list = zip(*dat)
        plt.plot(t_list, x_list, label=str(name))

plt.title("Motion of wave packet in accelerated Gaussian Box")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
# plt.savefig("accelerated_box")
plt.show()
