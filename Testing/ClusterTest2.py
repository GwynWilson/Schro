import numpy as np
import os
# cwd = os.getcwd()
# folder = "\\Dat"
# if not os.path.exists(cwd+folder):
#     try:
#         os.mkdir(cwd+folder)
#     except OSError:
#         print("whoops")
#
# arr = [0, 1, 2, 3, 4]
# np.savez_compressed("Dat/TestFile", arr)

dat = np.load("Dat/TestFile.npz")
print(dat["arr_0"])
