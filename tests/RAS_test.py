# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:39:49 2017

@author: vnarwade
"""
import pandas as pd
import numpy as np
from numpy import linalg
from random import randint

#import make_use_trial as trial

#Given Things

A = pd.DataFrame(np.array([[0.150,0.250,0.05],[0.20,0.05,0.4],[0.3,0.25,0.05]]))
x_1 = pd.DataFrame(np.array([[1200],[2500],[1400]]))
#Z_1 = pd.DataFrame(np.array([[98,72,75],[65,8,63],[88,27,44]]))
u_1 = pd.DataFrame(np.array([[780],[810],[850]]))
v_1 = pd.DataFrame(np.array([[740],[1270],[630]]))

#A = pd.DataFrame(np.array([[0.120,0.100,0.049],[0.210,0.247,0.265],[0.026,0.249,0.145]]))
#x_1 = pd.DataFrame(np.array([[421],[284],[283]]))
#Z_1 = pd.DataFrame(np.array([[98,72,75],[65,8,63],[88,27,44]]))
A1 = A

#A = trial.A
#Z_1 = trial.Z
#x_1 = trial.x

p = np.shape(A)[0]
#u_1 = Z_1.cumsum(axis = 1).iloc[:,-1]
#v_1 = Z_1.cumsum(axis = 0).iloc[-1, :]

x_1_cap = pd.DataFrame(np.zeros(shape = np.shape(A)))
u_1_cap = pd.DataFrame(np.zeros(shape = np.shape(A)))
v_1_cap = pd.DataFrame(np.zeros(shape = np.shape(A)))

for k in range (p):
        for j in range (p):
            if k == j:
                x_1_cap.iloc[k][j] = x_1.iloc[k]
                v_1_cap.iloc[k][j] = v_1.iloc[k]
                u_1_cap.iloc[k][j] = u_1.iloc[k]
                

i = pd.DataFrame(np.ones(shape = (np.shape(A)[0],1)))
I = pd.DataFrame(np.identity(np.shape(A)[0]))
Z = A.dot(x_1_cap)

u_0 = Z.dot(i)
v_0 = Z.transpose().dot(i)
u_0_cap = pd.DataFrame(np.zeros(shape = np.shape(A)))
v_0_cap = pd.DataFrame(np.zeros(shape = np.shape(A)))
r = pd.DataFrame(np.zeros(shape = np.size(u_1)))
s = pd.DataFrame(np.zeros(shape = np.size(u_1)))

iterations = 0
while not((np.allclose(r,I, rtol=0, atol=0.000001)) and 
          (np.allclose(s,I, rtol=0, atol=0.000001))):
    u_0 = Z.dot(i)
    v_0 = Z.transpose().dot(i)
    for k in range (p):
        for j in range (p):
            if k == j:
                u_0_cap.iloc[k][j] = u_0.iloc[k][0]
                v_0_cap.iloc[k][j] = v_0.iloc[k][0]
    r = u_1_cap.dot(linalg.inv(u_0_cap))
    
    
    A = r.dot(A)
    Z = A.dot(x_1_cap)
    u_0 = pd.DataFrame(Z.dot(i))
    v_0 = pd.DataFrame(Z.transpose().dot(i))
    for k in range (p):
        for j in range (p):
            if k == j:
                u_0_cap.iloc[k][j] = u_0.iloc[k][0]
                v_0_cap.iloc[k][j] = v_0.iloc[k][0]
    s = v_1_cap.dot(linalg.inv(v_0_cap))
        
    A = A.dot(s)
    Z = A.dot(x_1_cap)
    u_0 = pd.DataFrame(Z.dot(i))
    v_0 = pd.DataFrame(Z.transpose().dot(i))
    
    iterations += 1
    
    
#L = linalg.inv(I-A)

