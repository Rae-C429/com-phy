# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:35:05 2022

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

# set up finite matrix A
N = 31  
A = np.zeros((N**2, N**2))
for i in range (N**2):       
    if N > i :
        A[i, i] = 1
    elif (N * (N - 1)) <=  i :
        A[i, i] = 1
    elif (N <= i and i < N * (N - 1)) and (i % N == 0 or i % N == N - 1):
        A[i, i] = 1
    else : 
        A[i, i] = -4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
        A[i, i + N] = 1
        A[i, i - N] = 1
print("==============================================================================")
print("A\n", A)
print("==============================================================================")

# set finite plate charge and voltage
L = N - 1
h = L / (N - 1)
e = (N - 1) / N
top = int(N * 3 / 10)
bottom =int(N * 7 / 10)

lo = np.zeros((N, N))
for i in range (top, bottom + 1):
    lo[top, i] = e * (1./h**2)
    lo[bottom, i] = -e * (1./h**2)
print("==============================================================================")
print("lo\n", lo)
print("==============================================================================")

v = np.matmul(inv(A *(-h**2)), lo.reshape(N**2,1))
print("==============================================================================")
print("finite plate voltage \n", v.reshape(N, N))
print("==============================================================================")


x = np.arange(0, N, 1) 
y = np.arange(0, N, 1)
xv, yv = np.meshgrid(x, y)
xE = np.arange(0, N - 1, 1) 
yE = np.arange(0, N - 1, 1)
xvE, yvE = np.meshgrid(xE, yE)

# finite E-potential and E-field numerical picture
fig, ax = plt.subplots()
cf = ax.contourf(xv, yv, v.reshape(N, N), 10, cmap = 'bwr')
clb = fig.colorbar(cf)
clb.ax.set_title("$v$")

ax.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax.set_title("finite E-potential and E-field numerical")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()

# set up infinite matrix A

A = np.zeros((N**2, N**2))
for i in range (N**2):       
    if N > i :
        A[i, i] = 1
    elif (N * (N - 1)) <= i :
        A[i, i] = 1
    else : 
        A[i, i] = -4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
        A[i, i + N] = 1
        A[i, i - N] = 1
print("==============================================================================")
print("A\n", A)
print("==============================================================================")

# set infinite plate charge and voltage

lo = np.zeros((N, N))
lo[top, :] = e * (1./h**2)
lo[bottom, :] = -e * (1./h**2)
print("==============================================================================")
print("lo\n", lo)
print("lo\n", lo.reshape(N**2,1))
print("==============================================================================")
print("A", A)
v = np.matmul(inv(A *(-h**2)), lo.reshape(N**2,1))
print("==============================================================================")
print("infinite plate voltage \n", v.reshape(N, N))
print("==============================================================================")


# infinite E-potential and E-field numerical picture
fig, ax_in = plt.subplots()
cf = ax_in.contourf(xv, yv, v.reshape(N, N), 10, cmap = 'bwr')
clb_in = fig.colorbar(cf)
clb_in.ax.set_title("$v$")


ax_in.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax_in.set_title("infinite E-potential and E-field numerical")
ax_in.set_xlabel("$x$")
ax_in.set_ylabel("$y$")
plt.grid()
plt.show()
