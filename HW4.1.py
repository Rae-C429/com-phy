# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:11:37 2022

@author: pupss
"""
import numpy as np
import matplotlib.pyplot as plt

def black(wav, T):
    h = 6.62607515e-34
    hc = 1.98644586e-25
    k = 1.380649e-23
    a = 8.0 * np.pi * hc 
    b = hc / (wav * k * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity



def maximum(a):
    max_x = 0
    for i in range (len(a)):    
        if a[max_x]< a[i]:
            max_x = i
    return max_x

wav = np.arange(1e-9, 2.0e-6, 1e-10)

ints35 = black(wav, 3500)
print("3500 max is:", wav[maximum(ints35)])

ints40 = black(wav, 4000)
print("4000 max is:", wav[maximum(ints40)])

ints45 = black(wav, 4500)
print("4500 max is:", wav[maximum(ints45)])

ints50 = black(wav, 5000)
print("5000 max is:", wav[maximum(ints50)])

ints55 = black(wav, 5500)
print("5500 max is:",wav[ maximum(ints55)])


plt.plot(wav * 1e9, ints35, 'b-', label= 3500)
plt.plot(wav * 1e9, ints40, 'g-', label= 4000)
plt.plot(wav * 1e9, ints45, 'r-', label= 4500)
plt.plot(wav * 1e9, ints50, 'c-', label= 5000)
plt.plot(wav * 1e9, ints55, 'm-', label= 5500)
plt.xlabel("Wavelength ")
plt.ylabel("special energy density")
plt.legend()
plt.show()
