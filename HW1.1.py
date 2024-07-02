# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:33:40 2022

@author: pupss
"""
import numpy as np
a=np.arange(1,26,1)
print(a)
print("\n")
a=np.reshape(a,(5,5))
print(a)
print("\n")
for i in range (0,4):
    b=sum(a[i,:])
    print("row", i, "sum=", b)
print("\n")
a[2]=a[0,:]+a[2,:]
print("new matrix=\n")
print(a)