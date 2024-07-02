# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:59:56 2022

@author: pupss
"""

import numpy as np
import matplotlib.pyplot as plt
# 1.
print("1.")
print("------------------------------------------")
def H_atom(r):
    return ((1/(np.pi)**0.5)*((1/0.0529**1.5)*np.exp(-r/0.0529))*r)**2*4*np.pi
N = 10000
arear1 = 0
r=np.linspace(0, 100, N+1)
high = r[1] - r[0]
for i in range(0, N):
    top = H_atom(r[i])
    bottom = H_atom(r[i+1])
    ds = (top + bottom) * high / 2
    arear1 = arear1 + ds
print("trapezoid method ans1:", arear1)
for i in range(0, N):
    dx = H_atom(r[i+1]) - H_atom(r[i])
    ds = dx * high 
    arear1 = arear1 + ds
print("rectangle method ans1:", arear1)

# 2.
print("2.")
print("------------------------------------------")
def element(r):
    return np.exp(-3 * r / 2) * r**4
arear2 = 0
r=np.linspace(0, 100, N+1)
high = r[1] - r[0]
for i in range(0, N):
    top = element(r[i])
    bottom = element(r[i+1])
    ds = (top + bottom) * high / 2
    arear2 = arear2 +ds
print("trapezoid method ans2:", arear2 )
for i in range(0, N):
    dx = element(r[i+1]) - element(r[i])
    ds = dx * high 
    arear2 = arear2 + ds
print("rectangle method ans2:", arear2 )

