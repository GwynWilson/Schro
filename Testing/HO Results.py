import numpy as np

import matplotlib.pyplot as plt

import glob

alltxt = glob.glob("*.txt")
HO = []
exclude = ["0.005","-0.005","0.0001","-0.0001"]
exclude = ["0.0001","-0.0001","0.0005","-0.0005","0.001","-0.001"]

for i in alltxt:
    if i.startswith(str("HO_")):
        HO.append(i)

for i in HO:
    name = list(i.split("_"))[1][:-4]
    dat = np.loadtxt(i)
    a_list, x_list = zip(*dat)
    if name not in exclude:
        plt.plot(a_list, x_list, label=str(name))

plt.title("Amplitude vs Acceleration for Different Anharmonicity")
plt.xlabel("Acceleration")
plt.ylabel("Position")
plt.legend()
plt.savefig("HO_Anharmonicity_2")
plt.show()

