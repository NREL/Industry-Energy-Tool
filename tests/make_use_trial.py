# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:42:53 2017

@author: vnarwade
"""

import pandas as pd
import numpy as np
from numpy import linalg


make_file = 'make_monetary_final.xlsx'
use_file = 'use_monetary_final.xlsx'

use_table = pd.DataFrame(pd.read_excel(use_file,sheetname = 'use_monetary'))
use_table = use_table.set_index(['IO_code'])

make_table = pd.DataFrame(pd.read_excel(make_file,sheetname = 'make_monetary'))
make_table = make_table.set_index(['IO_code'])


rows_sort = list(use_table.index.astype(str))

U = pd.DataFrame(index = rows_sort, columns = rows_sort)
V = pd.DataFrame(index = rows_sort, columns = rows_sort)

for row in use_table.index:
    for column in use_table.columns:
        U.loc[str(row),str(column)] = use_table.loc[row,column]

for row in make_table.index:
    for column in make_table.columns:
        V.loc[str(row),str(column)] = make_table.loc[row,column]

#U = U.iloc[:71,:71]
#V = V.iloc[:71,:71]

i = np.ones((np.shape(U)[0],1))
I = np.identity(np.shape(U)[0])


x = pd.DataFrame(V.cumsum(axis = 1).iloc[:,-1])
x.columns = ['Total_Industry_Output']

x_cap = pd.DataFrame(columns = list(V.columns), index = list(V.index))
for row in list(V.index):
    for column in list(V.columns):
        if str(row) == str(column):
            x_cap.loc[row, column] = x.loc[row, 'Total_Industry_Output' ]
        
x_cap.fillna(0, inplace = True)

U.drop(['Total_Final_Demand'], axis = 1, inplace = True)
q_cap = pd.DataFrame(columns = list(U.columns), index = list(U.index))
q = pd.DataFrame(V.cumsum(axis = 0).iloc[-1,:])
q.columns = ['Total_Commodity_Output']


for row in list(U.index):
    for column in list(U.columns):
        if str(row) == str(column):
            q_cap.loc[row, column] = q.loc[row,'Total_Commodity_Output' ]
q_cap.fillna(0, inplace = True)


B = U.dot(linalg.inv(x_cap))
B.reset_index(inplace = True)
B.drop(['index'], axis = 1, inplace = True)

D = V.dot(linalg.inv(q_cap))
D.reset_index(inplace = True)
D.drop(['index'], axis = 1, inplace = True)

#x_cap.reset_index(inplace = True)
#x_cap.drop(['index'], axis = 1, inplace = True)
#x_cap = x_cap.transpose()
#x_cap.reset_index(inplace = True)
#x_cap.drop(['index'], axis = 1, inplace = True)
#x_cap = x_cap.transpose()

A = B.dot(D)
A.fillna(0, inplace = True)

Z = pd.DataFrame(np.dot(A, x_cap))
Z.columns = list(U.columns)
Z.index = list(U.index)

#writer = 'inter-industry.xlsx'
#Z.to_excel(writer,'inter-industry')

m = (I-A)
L = linalg.inv(I-A)

e = L.dot(q)

