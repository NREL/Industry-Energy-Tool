# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 10:56:50 2017

@author: vnarwade
"""

import pandas as pd
import numpy as np
from numpy import linalg

inter_industry_file = 'inter-industry-2014.xlsx'

Z = pd.read_excel(inter_industry_file, sheetname = 'monetary')
Z.set_index(['IO_code'], inplace = True)

Z_hybrid = pd.read_excel(inter_industry_file, sheetname = 'hybrid')
Z_hybrid.set_index(['IO_code'], inplace = True)

f_hybrid = Z_hybrid.f

x_hybrid = Z_hybrid.cumsum(axis = 1).iloc[:,-1]
x_hybrid.columns = ['Total_Industry_Output']

Z_hybrid.drop(['f'], axis = 1, inplace = True)


x_hybrid_cap = pd.DataFrame(columns = list(Z.columns.astype(str)), 
                     index = list(Z.index.astype(str)))
for row in list(x_hybrid.index):
    for column in list(Z.columns):
        if str(row) == str(column):
            x_hybrid_cap.loc[str(row), str(column)] = x_hybrid.loc[row]

x_hybrid_cap.fillna(0,inplace = True)

A_hybrid = Z_hybrid.dot(linalg.inv(x_hybrid_cap))

I = np.identity(A_hybrid.shape[0])
L_hybrid = linalg.inv(I-A_hybrid)

g = pd.read_excel(inter_industry_file, sheetname = 'use_energy_quad')
g.set_index(['IOCode'], inplace = True)
g.drop(['Name'],axis = 1, inplace = True)
g = g.cumsum(axis = 1).iloc[:,-1]
g.columns = ['Total Energy Consumption']

g_hybrid = pd.DataFrame(index = x_hybrid.index, columns = ['Total Energy Consumption'])
for row in list(g.index):
    g_hybrid.loc[row,'Total Energy Consumption'] = g[row]
g_hybrid.fillna(0, inplace = True)

G = pd.DataFrame(index = list(g.index.astype(str)), columns = list(Z.columns.astype(str)))
for row in list(G.index):
    for column in list(G.columns):
        if row == column:
            G.loc[row,column] = g.loc[int(row)] 
            
G.fillna(0, inplace = True)

alpha = (G.dot(linalg.inv(x_hybrid_cap))).dot(L_hybrid)
alpha.columns = [Z.columns]

A_hybrid = np.array(A_hybrid)

delta = (G.dot(linalg.inv(x_hybrid_cap))).dot(A_hybrid)
delta.columns = [Z.columns]