import numpy as np
# import seaborn
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
    # for w in w_list:
    #     dat = np.loadtxt(f"tanh_Tunneling w={w}.txt")

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.09, right=0.9, hspace=0.2, wspace=0.2)
    # fig.suptitle("Tanh Tunneling")
    print(V_scale,L_scale)

    dat1 = np.loadtxt(f"tanh_Tunneling w_a0={w_list[0]/a0}.txt")
    axs[0, 0].pcolormesh(V_scale / V_list, L_list / L_scale, dat1, cmap="gist_heat")
    axs[0, 0].set_title(fr"$\beta/l$ = {w_list[0]/L_scale:.2f}")

    dat2 = np.loadtxt(f"tanh_Tunneling w_a0={0.2}.txt")
    axs[0, 1].pcolormesh(V_scale / V_list, L_list / L_scale, dat2, cmap="gist_heat")
    axs[0, 1].set_title(fr"$\beta/l$={0.2*a0/L_scale }")

    dat3 = np.loadtxt(f"tanh_Tunneling w_a0={0.3}.txt")
    axs[1, 0].pcolormesh(V_scale / V_list, L_list / L_scale, dat3, cmap="gist_heat")
    axs[1, 0].set_title(fr"$\beta/l$ ={0.3*a0/L_scale:.1f}")

    dat4 = np.loadtxt(f"tanh_Tunneling w_a0={w_list[2]/a0}.txt")
    im = axs[1, 1].pcolormesh(V_scale / V_list, L_list / L_scale, dat4, cmap="gist_heat")
    axs[1, 1].set_title(fr"$\beta/l$ ={w_list[2]/L_scale:.1f}")

    fig.text(0.5, 0.05, r"$E/V_0$", va='center', ha='center')
    fig.text(0.04, 0.5, r"$L_t/l$", va='center', ha='center', rotation='vertical')

    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig("tanh_Subplots.png")
    plt.show()


V0 = 6.62607015 * 10 ** (-34) * 10 ** 3
L = 10 ** (-6)
m = 1.44316072 * 10 ** -25
hbar = 1.0545718 * 10 ** -34
bar_amp = 10 ** -30
E = bar_amp
l = np.sqrt(hbar ** 2 / (2 * m * bar_amp))

a0 = 5 * l

L_scale = l
V_scale = bar_amp

V_list = np.loadtxt("tanh_V_list.txt")
w_list = np.loadtxt("tanh_w_list.txt")
L_list = np.loadtxt("tanh_L_list.txt")
print(min(L_list))

plotFour()
