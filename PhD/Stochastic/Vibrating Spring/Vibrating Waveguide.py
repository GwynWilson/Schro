import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps


def oneRun(n, dt, x0, v0, a, b):
    xl = [x0]
    vl = [v0]
    t = 0
    tl = [0]
    x = x0
    v = v0
    xol = x0
    vol = x0
    for w in range(n - 1):
        t += dt
        tl.append(t)

        x += vol * dt
        xl.append(x)

        v += -a * xol * dt + b * np.sqrt(dt) * np.random.randn()
        # v += -a * x * dt
        vl.append(v)

        xol = x
        vol = v

    return tl, np.asarray(xl), np.asarray(vl)


def manyRun(n_runs, n, dt, x0, v0, a, b, energy=False):
    runsx = np.zeros([n_runs, n])
    runsv = np.zeros([n_runs, n])
    for i in range(n_runs):
        print(i)
        # if i % 100 == 0:
        #     print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx[i] = np.real(xdat)
        runsv[i] = np.real(vdat)
    else:
        return tl, runsx, runsv


def averageRun(n_runs, n, dt, x0, v0, a, b):
    runsx = np.zeros(n)
    runsv = np.zeros(n)
    runse = np.zeros(n)
    for i in range(n_runs):
        # print(i)
        if i % 10 == 0:
            print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx += xdat / n_runs
        runsv += vdat / n_runs
        runse += 0.5 * m * (vdat ** 2) + 0.5 * k * (xdat ** 2)
    return tl, runsx, runsv, runse


def average(list_):
    num, amount = np.shape(list_)
    comp = np.zeros(amount)
    for i in range(num):
        comp += list_[i] / num
    return comp


def gendat(n_runs, variations):
    all_listx = np.zeros((variations, n))  # Total lists
    all_listv = np.zeros((variations, n))
    all_liste = np.zeros((variations, n))
    t_l = 0
    for i in range(variations):
        dat = np.load(f"Dat/WaveguideDat{n_runs}_{i}.npz")
        tl = dat["tl"]
        xl = dat["xl"]
        vl = dat["vl"]
        el = dat["el"]
        all_listx[i] = xl
        all_listv[i] = vl
        all_listv[i] = el
        t_l = tl
    return t_l, all_listx, all_listv, all_liste


def energy(x, v, k, m):
    x = np.asarray(x)
    v = np.asarray(v)
    return 0.5 * m * (v ** 2) + 0.5 * k * (x ** 2)


def expectedE(t, k, m, x0, v0, sig):
    t = np.asarray(t)
    return 0.5 * k * x0 ** 2 + 0.5 * m * v0 ** 2 + 0.5 * sig ** 2 * t / m
    # return 0.5 * k * x0 ** 2 + 0.5 * m * v0 ** 2 + sig ** 2 * t / (2*m**2)


def manyRun(n_runs, n, dt, x0, v0, a, b, energy=False):
    runsx = np.zeros([n_runs, n])
    runsv = np.zeros([n_runs, n])
    runse = np.zeros([n_runs, n])
    for i in range(n_runs):
        # print(i)
        if i % 100 == 0:
            print(i)
        tl, xdat, vdat = oneRun(n, dt, x0, v0, a, b)
        runsx[i] = np.real(xdat)
        runsv[i] = np.real(vdat)
        if energy:
            runse[i] = 0.5 * m * np.asarray(vdat) ** 2 + 0.5 * k * np.asarray(xdat) ** 2

    if energy:
        return tl, runsx, runsv, runse
    else:
        return tl, runsx, runsv


def runsData(data):
    runs, l = np.shape(data)
    average = np.zeros(l)
    variance = np.zeros(l)
    for i in range(l):
        if i % 10000 == 0:
            print(i)
        dslice = data[:, i]
        average[i] = np.mean(dslice)
        variance[i] = np.var(dslice)
    return average, variance


def expectedVar(t, sig, w):
    t = np.asarray(t)
    return sig ** 2 / (2 * w ** 2 * m ** 2) * (t - np.sin(2 * w * t) / (2 * w))

def poly(x,a):
    return a*x


d = 0.004
v = 0.01
t = d / v
print("Final time",t)
n = 500000
dt = t / n
print("dt", dt)

m = 1.44 * 10 ** (-25)
w = 6.1 * 10 ** 2
k = w ** 2 * m

print("t0", w ** (-1))

sig = m / 100

x0 = 0
v0 = 0
# print("t0",1/w)
# print(dt)


## One Run
# tl, xl, vl = oneRun(n, dt, x0, v0, w ** 2, sig/m)
# plt.plot(tl, xl)
# plt.show()
#
# plt.plot(tl, vl)
# plt.show()
#
# plt.plot(tl,energy(xl,xl,k,m))
# plt.show()


# n_runs = 100
# tl, xl, vl = manyRun(n_runs, n, dt, x0, v0, w ** 2, sig/m)
# for i in xl:
#     plt.plot(tl, i)
# plt.show()
# for i in vl:
#     plt.plot(tl, i)
# plt.show()


###### Average Run

# n_runs = 1000
# tl, xl, vl, el = averageRun(n_runs, n, dt, x0, v0, w ** 2, sig/m)
# np.savez_compressed(f"Dat/WaveguideDat{n_runs}_dt{dt:.0E}", tl=tl, xl=xl, vl=vl ,el=el)
# plt.plot(tl,xl)
# plt.show()
#
# plt.plot(tl,vl)
# plt.show()
#
# plt.plot(tl,el)
# plt.show()
# dat = np.load(f"Dat/WaveguideDat{n_runs}.npz")
# tl = dat["tl"]
# xl = dat["xl"]
# vl = dat["vl"]
# el = dat["el"]
#
# plt.plot(tl, xl)
# plt.title(f"Waveguide X position, n={n_runs}, dt={dt:.0E}")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.savefig("Waveguide Position")
# plt.show()
#
# plt.plot(tl, vl)
# plt.title(f"Waveguide Velocity, n={n_runs}, dt={dt:.0E}")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m s^(-1))")
# plt.savefig("Waveguide Velocity")
# plt.show()
#
# plt.plot(tl, el)
# plt.title(f"Waveguide Energy, n={n_runs}, dt={dt:.0E}")
# plt.xlabel("Time (s)")
# plt.ylabel("Energy (J)")
# plt.savefig("Waveguide Energy")
# plt.show()

##### Average Run_variations
n_runs = 1000
# variations = 10
# for i in range(variations):
#     print("---------------------", i)
#     tl, xl, vl, el = averageRun(n_runs, n, dt, x0, v0, w ** 2, sig / m)
#     np.savez_compressed(f"Dat/WaveguideDat{n_runs}_{i}_dt{dt:.0E}", tl=tl, xl=xl, vl=vl, el=el)

# variations = 10
# compx = np.zeros(n)
# compv = np.zeros(n)
# compe = np.zeros(n)
# for i in range(variations):
#     dat = np.load(f"Dat/WaveguideDat{n_runs}_{i}_dt{dt:.0E}.npz")
#     tl = dat["tl"]
#     xl = dat["xl"]
#     vl = dat["vl"]
#     el = dat["el"]
#     compx += xl / variations
#     compv += vl / variations
#     compe += el / variations
# np.savez_compressed(f"Dat/WaveguideDat{n_runs * variations}_dt{dt:.0E}", tl=tl, xl=compx, vl=compv, el=compe)
# plt.plot(tl, compx)
# plt.show()
# plt.plot(tl, compv)
# plt.show()

######### Progression
# n_runs = 1000
# variations = 10
#
# all_x = []
# for i in range(variations):
#     tl, xlist, vlist, elist = gendat(n_runs, i)
#     if i > 0:
#         all_x.append(average(xlist))
#     else:
#         all_x.append(xlist)
#
# pltnums = [1, 3, 7]
# for i in pltnums:
#     plt.plot(tl, all_x[i], label=f"{n_runs * (i + 1)}")
# plt.legend()
# plt.show()

##### Manydat
# n_list = [100, 1000, 10000]
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# for i in n_list:
#     dat = np.load(f"Dat/WaveguideDat{i}_dt{dt:.0E}.npz")
#     tl = dat["tl"]
#     xl = dat["xl"]
#     vl = dat["vl"]
#     el = dat["el"]
#     ax1.plot(tl, xl)
#     ax2.plot(tl,vl)
# fig.suptitle(f"Waveguide Time averaging, dt={dt:.0E}")
# ax1.ticklabel_format(style="sci",scilimits=(0,0))
# ax2.ticklabel_format(style="sci",scilimits=(0,0))
# ax1.set_ylabel("x")
# ax2.set_ylabel("v")
# ax2.set_xlabel("t")
# # plt.savefig("Waveguide Time Averaging")
# plt.show()

##### ManyDat Energy
# n_list = [10, 100, 1000, 10000]
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
#
# for i in n_list:
#     dat = np.load(f"Dat/WaveguideDat{i}_dt{dt:.0E}.npz")
#     tl = dat["tl"]
#     xl = dat["xl"]
#     vl = dat["vl"]
#     el = dat["el"]
#     energ_expect = expectedE(tl, k, m, 0, 0, sig)
#     ax1.plot(tl, el)
#     ax2.plot(tl, energ_expect - el)
#
# ax1.plot(tl, energ_expect,color="k")
# plt.show()


####### Curve Fitting
dat = np.load(f"Dat/WaveguideDat{10000}_dt{dt:.0E}.npz")
tl = dat["tl"]
xl = dat["xl"]
vl = dat["vl"]
el = dat["el"]


popt, pcov = curve_fit(poly, tl, el)
print("grad",popt[0])
print("m",m)
print("sig",sig)
print("sig^2/m%2",sig**2/m**2)
print("Grad pred",sig**2/(2*m))
print(popt[0]/sig)


# plt.plot(tl,el)
# plt.plot(tl,poly(tl,popt[0]))
# plt.show()

###### Pos/Var
##Gather


# n_list = [10,100,1000]
# for n_runs in n_list:
#     print("-----------",n_runs)
#     tl, xdat, vdat = manyRun(n_runs, n, dt, 0, 0, w ** 2, sig / m)
#     xav, xvar = runsData(xdat)
#     vav, vvar = runsData(vdat)
#     np.savez_compressed(f"Dat/WaveguideDat_Var_x_{n_runs}_smallerdt",av=xav,var=xvar)
#     np.savez_compressed(f"Dat/WaveguideDat_Var_v_{n_runs}_smallerdt", av=vav, var=vvar)


######Analyse
# n_list = [10, 100, 1000]
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# tl = np.asarray([i * dt for i in range(n)])
# for i in n_list:
#     xdat = np.load(f"Dat/WaveguideDat_Var_x_{i}.npz")
#     xav = xdat["av"]
#     xvar = xdat["var"]
#     ax1.plot(tl, xav)
#
#     vdat = np.load(f"Dat/WaveguideDat_Var_v_{i}.npz")
#     vav = vdat["av"]
#     vvar = vdat["var"]
#     ax2.plot(tl, xvar, label=f"{i}")
# fig.suptitle(f"Numerical Testing dt={dt:.2E}")
# ax1.ticklabel_format(style="sci",scilimits=(0,0))
# ax2.ticklabel_format(style="sci",scilimits=(0,0))
# ax1.set_ylabel("Position (m)")
# ax2.set_ylabel("Variance")
# ax2.set_xlabel("Time (s)")
# ax2.legend(loc=2)
# ax2.plot(tl, expectedVar(tl, sig, w), color="k")
# # plt.savefig(f"Waveguide_posvar_dt{dt:.0E}")
# plt.show()

######## Additional length
# tl, xl, vl = oneRun(n, dt, x0, v0, w ** 2, sig / m)
#
# plt.plot(tl, vl)
# plt.show()
#
# totv = np.sqrt(v**2 + vl**2)
#
# d_sim = simps(totv,tl)
# print("Horizontal distance",np.sqrt(d**2))
# print("Total distance",d_sim)
