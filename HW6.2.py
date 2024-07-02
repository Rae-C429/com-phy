# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:05:45 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

def E(ka):
    t = -1
    E = 2 * t * np.cos(ka)
    return E

N = 5

# set matrix
Hmn = np.zeros((N, N), dtype = float)
Hmn[1][0] = -1
Hmn[N - 2][N - 1] = -1
for m in range(1, N - 1):
    Hmn[m - 1][m] = -1
    Hmn[m + 1][m] = -1
print(Hmn) 
   


# solve eigen
value, vector = linalg.eig(Hmn)
idx = np.argsort(value)
value = value[idx]
value = np.concatenate((np.flip(value, 0), value), axis=0)
print(value)
x = np.linspace(-np.pi, np.pi, N * 2)
inf_x = np.linspace(-np.pi, np.pi, N * 10)
print(x)
fig1, ax1 = plt.subplots()
ax1.plot(x, value, 'bo', label="finite")
ax1.plot(inf_x , E(inf_x), 'r-', label="infinite")
ax1.set_title("tight binding model")
ax1.set_ylabel("$E$")
ax1.set_xlabel("$ka$")
ax1.legend()
plt.grid()
plt.show