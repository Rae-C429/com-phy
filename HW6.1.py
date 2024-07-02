# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:41:20 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
N = 300
N_grid = 2 * N + 1
H = np.zeros((N_grid, N_grid), dtype = float)
width = 10
dx = 0.1

# finite
for i in range(0, N_grid):

    for j in range(0, N_grid):
        if   j == i - 1:
            H[i,j] = - 0.5 / (dx**2)
            
        elif i == j:
            x = dx * (i - N)
            
            if   x > -width / 2  and  x < width / 2:
                H[i][j] = 1.0 / (dx**2) 
                
            else :
                H[i][j] = 1.0 / (dx**2) + 0.3
            
        elif j == i + 1:
            H[i][j] = - 0.5 / (dx**2)
print("finite")
print("======================================================")
print("H\n", H)
value, vector = linalg.eig(H)
# print(value[0])
# print(vector[:, 0])
# print(value)
idx = np.argsort(value)
value = value[idx]
vector = vector[:, idx] * 0.67

print("value[0]\n", value[0])
print("vector[:, 0]\n", vector[:, 0])

nmax = 5
nsho = np.linspace(0, nmax, nmax + 1)
# print("n = \n", nsho)
# print("E_n\n", value[0 : nmax])

fig1, ax1 = plt.subplots()
#ax1.plot(nsho, nsho, label = "analytical")
ax1.plot(nsho + 1, value[0 : nmax + 1] * 10000, 'o', label = "numerical")
ax1.set_xlabel("n")
ax1.set_ylabel('$E_n/\hbar\omega_0$')
ax1.set_title("finite: eigen energies")
ax1.set_xticks(np.linspace(1, nmax + 1, nmax * 2 + 1))
# ax1.set_yticks(np.arange(0, N/10, 1))

plt.grid(linewidth = 0.5)
ax1.legend()


fig2, ax2 = plt.subplots()
x = np.linspace(0, N_grid - 1, N_grid)
x = x -N
x = x * dx
ax2.plot(x, vector[:, 0], label = "$\psi_{n=0}(x)$")
# ax2.plot(x, vector[:, 1], label = "$\psi_{n=0}(x)$")
# ax2.plot(x, vector[:, 2], label = "$\psi_{n=0}(x)$")
ax2.set_ylabel("wave function")
ax2.set_xlabel("position(x/$\lambda_0$)")
ax2.set_title("finite: The wave of the lowest eigen state: n = 1")
ax2.legend()
# plt.show()

# print("------------------------------------------------------")
# # infinite

# for i in range(0, N_grid):
#     for j in range(0, N_grid):
#         if   j == i - 1:
#             H[i,j] = - 0.5 / (dx**2)
            
#         elif i == j:
#             x = dx * (i - N)
            
#             if   x > -width / 2  and  x < width / 2:
#                 H[i][j] = 1.0 / (dx**2) + 0
                
#             else :
#                 H[i][j] = 1.0 / (dx**2) + 10000
            
#         elif j == i + 1:
#             H[i][j] = - 0.5 / (dx**2)
# print("finite")
# print("======================================================")
# print("H\n", H)
# value, vector = linalg.eig(H)
# idx = np.argsort(value)
# value = value[idx]
# vector = vector[:, idx] * 0.67
# print("value[0]\n", value[0])
# print("vector[:, 0]\n", vector[:, 0])

# fig3, ax3 = plt.subplots()
# #ax1.plot(nsho, nsho, label = "analytical")
# ax3.plot(nsho + 1, value[0 : nmax + 1] * 20, 'o', label = "numerical")
# ax3.set_xlabel("n")
# ax3.set_ylabel('$E_n/\hbar\omega_0$')
# ax3.set_title("infinite: eigen energies")
# ax3.set_xticks(np.linspace(1, nmax + 1, nmax * 2 + 1))
# # ax3.set_yticks(np.arange(0, N/10, 1))

# plt.grid(linewidth = 0.5)
# ax1.legend()


# fig4, ax4 = plt.subplots()
# x = np.linspace(0, N_grid - 1, N_grid)
# x = x -N
# x = x * dx
# ax4.plot(x, -vector[:, 0], label = "$\psi_{n=0}(x)$")
# ax4.plot(x, -vector[:, 1], label = "$\psi_{n=0}(x)$")
# ax4.plot(x, -vector[:, 2], label = "$\psi_{n=0}(x)$")
# ax4.set_ylabel("wave function")
# ax4.set_xlabel("position(x/$\lambda_0$)")
# ax4.set_title("infinite: The wave of the lowest eigen state: n = 1")
# ax4.legend()
plt.show()

