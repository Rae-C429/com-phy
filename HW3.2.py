# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 00:04:35 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
a = 0
Ei = np.empty(shape=9)
for i in range(9):
    Ei[i]= a
    a += 25
g = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])

Ei_lag = np.linspace(0, 200, 200+1)


g_lag = 0
n = 9

for i in range(n):
    
    p = 1
    
    for j in range(n):
        if i != j:
            p = p * (Ei_lag - Ei[j])/(Ei[i] - Ei[j])
    
    g_lag = g_lag + p * g[i]

f_interp = CubicSpline(Ei, g)
Ei_c = np.linspace(0, 200, 200+1)
g_c = f_interp(Ei_c)
fig, ax = plt.subplots()
ax.plot(Ei, g, 'o', markerfacecolor='none', markeredgecolor='r', label= 'data')
ax.plot(Ei_lag, g_lag, 'b--', label= 'Lagrange')
ax.plot(Ei_c, g_c, 'r-', label= 'Cubic')
ax.legend()