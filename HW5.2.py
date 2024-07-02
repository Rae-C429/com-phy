# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:55:22 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# set electric field
def Efx(v):
    Efx = np.zeros((N - 1, N - 1))
    for i in range (N - 2):
        for j in range (N -2):
            Efx[i, j] = (v[i, j + 1] - v[i, j] + v[i + 1, j + 1] - v[i + 1, j]) * (1 / (2 * h))

    return Efx
        
def Efy(v):
    Efy = np.zeros((N - 1, N - 1)) 
    for i in range (N - 2):
        for j in range (N - 2):
            Efy[i, j] = (v[i, j + 1] - v[i + 1, j + 1] + v[i, j] - v[i + 1, j]) * (-1 / (2 * h))

    return Efy

N = 9


# set voltage
top = int(N * 3 / 10)
bottom =int(N * 7 / 10)

v = np.zeros((N, N))
for i in range (top, bottom + 1):
    v[top, i] = 1
    v[bottom, i] = -1


print(v)
h = 1
k = 0
while k < 45:
    v_new = np.zeros((N, N))
    for i in range (1, N - 1):
        for j in range (1, N - 1):
            v_new[i, j] = (1. / 4.) * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1]
                        + v[i, j - 1])
    v = v_new
    k = k + 1
            
print(v)


x = np.arange(0, N, 1) 
y = np.arange(0, N, 1)
xv, yv = np.meshgrid(x, y)
fig, ax = plt.subplots()
cf = ax.contourf(xv, yv, v.reshape(N, N), 10, cmap = 'bwr')
clb = fig.colorbar(cf)
clb.ax.set_title("$v$")

xE = np.arange(0, N - 1, 1) 
yE = np.arange(0, N - 1, 1)
xvE, yvE = np.meshgrid(xE, yE)
ax.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax.set_title("E-potential and E-field numerical")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()