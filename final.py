# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:23:12 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg
N = 35  

''' set charge'''
L = N - 1
h = L / (N - 1)
e = (N - 1) / N

''' set top and bottom plates position '''
bottom = int(N * 3 / 10)
top =int(N * 7 / 10)
a = int(N / 7)
lo = np.zeros((N, N))

''' bottom capacitor  plate '''
lo[bottom, a: N - a] =  e * (1./h**2) * 0

''' top capacitor  plate '''
# first
lo[top, a : a + a] = e * (1./h**2) * 40

# second
m = int(N / 2) - 1
lo[top, m: m + a] = e * (1./h**2) * 30

# third
lo[top, N - a - a: N - a] = e * (1./h**2) * 40 # N - a
print("lo\n", lo)


''' matrix A '''
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
        
print("A\n", A)


''' set voltage '''
v = np.matmul(inv(A *(-h**2)), lo.reshape(N**2,1))
print("plate voltage :\n", v.reshape(N, N))

v_top =v.reshape(N, N)[top, :]
print("top voltage: \n", v_top)
x = np.arange(0, N, 1) 
y = np.arange(0, N, 1)
xv, yv = np.meshgrid(x, y)
xE = np.arange(0, N - 1, 1) 
yE = np.arange(0, N - 1, 1)
xvE, yvE = np.meshgrid(xE, yE)


# finite E-potential and E-field numerical picture
fig1, (ax, bx) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
cf = ax.contourf(xv, yv, v.reshape(N, N), 10) # cmap = 'rainbow_r'
clb = fig1.colorbar(cf)
clb.ax.set_title("$v$")
ax.set_title("finite E-potential and E-field numerical")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
bx.plot(x, v_top)
bx.set_title("top voltage")
bx.set_xlabel("x")
bx.set_ylabel("voltage")
bx.grid()
plt.show()


'''wavefunction'''
width = 10
dx = 0.1
def wave(N):
    H = np.zeros((N, N), dtype = float)
    # finite
    for i in range(0, N):
        for j in range(0, N):
            if   j == i - 1:
                H[i,j] = - 0.5 / (dx**2) * 1
            
            elif i == j:
                H[i][j] = 1.0 / (dx**2) + v_top[i] 
            
            elif j == i + 1:
                H[i][j] = - 0.5 / (dx**2) 
    return(H)
print(wave(N))
value, vector = linalg.eig(wave(N))
idx = np.argsort(value)
value = value[idx]
vector = vector[:, idx] * 0.67

print("value[0]\n", value[0])
print("vector[:, 0]\n", vector[:, 0])

x = np.linspace(0, N - 1, N)
fig2, (cx, dx) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
cx.plot(np.linspace(0, 10, 10), value[0: 10], 'o')
cx.set_xlabel("n")
cx.set_ylabel('$E_n/\hbar\omega_0$')
cx.set_title("finite: eigen energies")
cx.grid(linewidth = 0.5)
dx.plot(x, - vector[:, 4])
dx.plot(x, - vector[:, 5])
dx.set_ylabel("wave function")
dx.set_xlabel("position(x/$\lambda_0$)")
dx.set_title("finite: The wave of the lowest eigen state: n = 1 $\psi_{n=0}(x)$")
dx.grid(linewidth = 0.5)
plt.show()