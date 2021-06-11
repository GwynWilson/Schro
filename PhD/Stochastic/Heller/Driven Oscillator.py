import numpy as np
from Heller import Heller


def derivs(t, current, args, eta, dt):
    w, m, hbar, wp = args
    x = current[1]
    v = -w ** 2 * current[0] + w ** 2 * eta
    a = -2 * current[2] ** 2 / m - m * w ** 2 / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * (current[0] - eta) ** 2 / 2
    return x, v, a, g


def derivs2(t, current, args, eta, dt):
    w, m, hbar, wp, sig = args
    x = current[1]
    v = -w ** 2 * current[0] + sig * w ** 2 * eta
    a = -2 * current[2] ** 2 / m - m * w ** 2 / 2
    g = 1j * hbar * current[2] / m + m * current[1] ** 2 / 2 - m * w ** 2 * (current[0] - sig*eta) ** 2 / 2
    return x, v, a, g


def expect(t, args, init):
    t = np.asarray(t)
    w, m, hbar, wp = args
    x0, v0, a0, g0 = init
    A = w ** 2 * wp / (w ** 2 - wp ** 2)
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w + A * (np.sin(wp * t) / wp - np.sin(w * t) / w)
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t) + A * (np.cos(wp * t) - np.cos(w * t))
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))

    g_ex = g0 - 0.5 * hbar * w * t - 0.5 * m * w ** 2 * (t / 2 - np.sin(2 * wp * t) / (4 * wp)) \
            +m*w*A *((w/wp)*(t / 2 - np.sin(2 * wp * t) / (4 * wp)) - (wp*np.cos(wp*t)*np.sin(w*t) - w*np.cos(w*t)*np.sin(wp*t))/(w**2-wp**2))\
            +0.5*m*A**2*(0.5*t*(1-w**2/wp**2) +np.sin(2*w*t)/(2*w) +(1+w**2/wp**2)*np.sin(2*wp*t)/(4*wp)+(2/(w**2-wp**2)) * \
                         (wp*np.cos(w*t)*np.sin(wp*t)- w**2*np.cos(w*t)*np.sin(wp*t)/wp))

    return x_ex, v_ex, a_ex, g_ex


def expect2(t, args, init):
    t = np.asarray(t)
    w, m, hbar, wp, sig = args
    x0, v0, a0, g0 = init
    A = w ** 2 * wp * sig / (w ** 2 - wp ** 2)
    x_ex = x0 * np.cos(w * t) + v0 * np.sin(w * t) / w + A * (np.sin(wp * t) / wp - np.sin(w * t) / w)
    v_ex = v0 * np.cos(w * t) - w * x0 * np.sin(w * t) + A * (np.cos(wp * t) - np.cos(w * t))
    temp = 0.5 * m * w
    cot = 1 / np.tan(w * t)
    a_ex = -temp * ((temp - a0 * cot) / (a0 + temp * cot))

    fts = w**2*(t/2 -0.25*np.sin(2*wp*t)/wp)

    xft = w*A*((w/wp)*(t/2 - np.sin(2*wp*t)/(4*wp)) - (wp*np.cos(wp*t)*np.sin(w*t) - w*np.cos(w*t)*np.sin(wp*t))/(w**2-wp**2))\
            +w**2*x0*(-wp+wp*np.cos(w*t)*np.cos(wp*t)+w*np.sin(w*t)*np.sin(wp*t))/(w**2-wp**2)\
            +v0*w*(wp*np.cos(wp*t)*np.sin(w*t)-w*np.cos(w*t)*np.sin(wp*t))/(w**2-wp**2)

    vsxs1 = A**2*(0.5*t*(1-w**2/wp**2) +np.sin(2*w*t)/(2*w)+(1+w**2/wp**2)*np.sin(2*wp*t)/(4*wp)\
                  +(2/(w**2-wp**2))*(wp-w**2/wp)*np.cos(w*t)*np.sin(wp*t))

    vsxs2 = (v0**2-w**2*x0**2)*(0.5*np.sin(2*w*t)/w) +x0*v0*(np.cos(2*w*t)-1)

    vsxs3 = -2*x0*w*A*(0.5*(np.cos(2*w*t)-1)/w + (1/(w**2-wp**2))*((-wp+w**2/wp)*np.sin(w*t)*np.sin(wp*t)))

    vsxs4 = 2*v0*A*(-0.5*np.sin(2*w*t)/w +1/(w**2-wp**2) *(w**2/wp - wp)*np.cos(w*t)*np.sin(wp*t))

    g_ex = g0 - 0.5 * hbar * w * t +0.5*m*(vsxs1+vsxs2+vsxs3+vsxs4)+m*sig*xft-0.5*m*fts*sig**2
    return x_ex, v_ex, a_ex, g_ex


def drive(t, wp):
    t = np.asarray(t)
    return np.sin(wp * t)


n = 1000
dt = 0.001
t = [i * dt for i in range(n)]

w = 10
wp = 2 * w
m = 1
hbar = 1
sig = 2
args = (w, m, hbar, wp,sig)

a0 = 1j * m * w / 2
init = [0.5, 5, a0, 0]

driving = drive(t, wp)

solver = Heller(n, dt, init, derivs2)
solver.rk4(args, noise=driving)
solver.plotBasic(expected=expect2)
