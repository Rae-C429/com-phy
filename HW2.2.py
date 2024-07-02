# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:55:10 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt

N = 51
h = 10 / (N-1) 

x = np.zeros(51)
a = 0
for i in range (51):
    x[i] = a
    a += 0.2
print("x", x)
print("-------------------------------------------------------------------------------")

v = 1 -np.exp(-x)
print("v:\n", v)
print("-------------------------------------------------------------------------------")

dv = np.zeros((N), dtype = float)
for i in range(1, N-2):
    dv[i] = (v[i + 1] - v[i - 1]) / (2 * h)
print("dv:\n", dv)

fig, ax = plt.subplots()
ax.set_title('dVy / d$\\tau$ = 1 - Vy : three-point formula')
plt.xlabel('t')
plt.ylabel('$V_t$')
ax.plot(x[1 : N-2], v[1 : N-2], 'r-', label = 'ana.')
ax.plot(x[2 : N-2], 1 - dv[2 : N-2], 'bs', label = 'num.')
ax.legend(loc = 'lower right')
plt.show()