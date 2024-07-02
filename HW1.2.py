# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:13:48 2022

@author: pupss
"""

import numpy as np
from numpy.linalg import inv
A = np.array([[-1, 1, -1, 0, 0, 0],  #A
                    [4, 2, 0, 0, 0, 0],   #top       
                    [1, -1, 0, 1, 0, 0],  #B
                    [0, 0, 1, 0, -1, 1],  #c
                    [0, 2, 0, 0, 4, 0],   #middel
                    [0, 0, 0, 0, 4, 5],   #bottom
                    [0, 0, 0, -1, 1, -1]])#D
b = np.array([[0], #A
                [8], #top
                [0], #B
                [0], #C
                [0], #middel
                [10],#bottom
                [0]])#D
n = 6 
for i in range(0,  n-1): 
    for j in range(i + 1, n+1): 
        if A[j, i] != 0.0:
            lam = A[j, i] / A[i, i]
            A[j, i : n] = A[j, i : n]- lam * A[i, i  :n]
            b[j] = b[j] - lam * b[i]
print("Gaussian Elimination")
print("=========================")
print("orignal")
print("-------------------------")
print(" A =\n",A)
print("b =\n", b)
print("=========================")
print("modify")
print("-------------------------")
A = A[0 : n,0 : n] 
b =  b[0 : n] 
print("A =\n",A)
print("b =\n", b)
print("=========================")
x=np.zeros((6,1))
x[n-1] =b[n-1] / A[n-1,n-1] 
for i in range(n-2,-1,-1): 
    x[i]=(b[i] - np.dot(A[i, i + 1 : n],x[i + 1 : n])) / A[i, i]
print("x =\n",x)
print("=========================")
print("inverse")
x = np.dot(inv(A), b)
print("x =\n",x)