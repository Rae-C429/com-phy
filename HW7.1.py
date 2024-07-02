# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:13:25 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
m = 1
gravity = 9.8
k = 0.1
Vt = m * gravity / k
def g(y, t):
    g = (m * gravity - k * y) * m 
    return g
def ysol(t):
    y = Vt * (1 - np.exp(- k * t / m ))
    return y
h = 1
N = 50
yi = np.zeros(N)
gi = np.zeros(N)
ti = np.arange(N) * h
ai = np.zeros(N)

t0 = ti[0]
yi[0] = Vt * (1 - np.exp(- k * t0 / m ))
for i in range(0, N - 1):
    gi[i] = g(yi[i], ti[i])
    yi[i + 1] = yi[i] + h * gi[i]



ai = ysol(ti)
fig1, ax1 = plt.subplots()
ax1.plot(ti, yi, "o", label = "Eular")
ax1.plot(ti, ai, label = "Solution")
plt.xlabel("$t$")
plt.ylabel("$v$")
ax1.text(20, 4.5, "$dv_y/dt= mg - k \cdot v_y$", fontsize = 'large',)
ax1.legend()

def RKTwo():
    yi[0] = Vt * (1 - np.exp(- k * t0 / m ))
    for i in range(N - 1):
        k1 = h * g(yi[i], ti[i])
        k2 = h * g(yi[i] + h, ti[i] + k1)
        rk2 = yi + h * (k1 + k2)
    return rk2

fig2, ax2 = plt.subplots()
ax2.plot(ti, RKTwo(), "ro", label = "RKTwo")
plt.xlabel("$t$")
plt.ylabel("$v$")
ax2.text(20, 4.5, "$dv_y/dt= mg - k \cdot v_y$", fontsize = 'large')
ax2.plot(ti, ai, label = "Solution")
ax2.legend()
plt.show()