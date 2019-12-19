import numpy as np
import seaborn
import matplotlib.pyplot as plt


def plotAll():
    for w in w_list:
        dat = np.loadtxt(f"tanh_Tunneling w={w}.txt")
        a = plt.pcolormesh(V_list / V_scale, L_list / L_scale, dat, cmap="gist_heat")
        plt.colorbar(a)
        plt.title(f"Tanh Barrier Tunneling w={w}")
        plt.xlabel("V0/E")
        plt.ylabel("Barrier Length (micrometers)")
        plt.show()


def plotFour():
    for w in w_list:
        dat = np.loadtxt(f"tanh_Tunneling w={w}.txt")

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.09, right=0.9, hspace=0.2, wspace=0.2)
    fig.suptitle("Tanh Tunneling")

    dat1 = np.loadtxt(f"tanh_Tunneling w={w_list[0]}.txt")
    axs[0, 0].pcolormesh(V_list / V_scale, L_list / L_scale, dat1, cmap="gist_heat")
    axs[0, 0].set_title(f"w = {w_list[0]}")

    dat2 = np.loadtxt(f"tanh_Tunneling w={w_list[3]}.txt")
    axs[0, 1].pcolormesh(V_list / V_scale, L_list / L_scale, dat2, cmap="gist_heat")
    axs[0, 1].set_title(f"w={w_list[2]}")

    dat3 = np.loadtxt(f"tanh_Tunneling w={w_list[-2]}.txt")
    axs[1, 0].pcolormesh(V_list / V_scale, L_list / L_scale, dat3, cmap="gist_heat")
    axs[1, 0].set_title(f"w={w_list[-2]}")

    dat4 = np.loadtxt(f"tanh_Tunneling w={w_list[-1]}.txt")
    im = axs[1, 1].pcolormesh(V_list / V_scale, L_list / L_scale, dat4, cmap="gist_heat")
    axs[1, 1].set_title(f"w={w_list[-1]}")

    fig.text(0.5, 0.05, "V0/E", va='center', ha='center')
    fig.text(0.04, 0.5, "Barrier Length (micrometers)", va='center', ha='center', rotation='vertical')

    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.show()


V_scale = 6.62607015 * 10 ** (-34) * 10 ** 3
L_scale = 10 ** (-6)

V_list = np.loadtxt("tanh_V_list.txt")
w_list = np.loadtxt("tanh_w_list.txt")
L_list = np.loadtxt("tanh_L_list.txt")

plotFour()
