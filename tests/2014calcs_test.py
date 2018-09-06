# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:18:14 2017

@author: vnarwade
"""

import pandas as pd

use2014 = pd.read_excel('use_monetary_final.xlsx', sheetname='2014')
use2014 = use2014.set_index(['IO_code'])

use2013 = pd.read_excel('use_monetary_final.xlsx', sheetname='2013')
use2013 = use2013.set_index(['IO_code'])

use2013_14 = pd.DataFrame(index = list(use2014.index.astype(str)), 
                         columns = list(use2014.columns.astype(str)))

for row in list(use2013.index):
    for column in list(use2013.columns):
        use2013_14.loc[str(row),str(column)] = use2013.loc[row,column]
        
#writer = 'use2013_14.xlsx'
#use2013_14.to_excel(writer,'2013')

make2014 = pd.read_excel('make_monetary_final.xlsx', sheetname='2014')
make2014 = make2014.set_index(['IO_code'])

make2013 = pd.read_excel('make_monetary_final.xlsx', sheetname='2013')
make2013 = make2013.set_index(['IO_code'])

make2013_14 = pd.DataFrame(index = list(use2014.index.astype(str)), 
                         columns = list(use2014.columns.astype(str)))

make2014_14 = pd.DataFrame(index = list(use2014.index.astype(str)), 
                         columns = list(use2014.columns.astype(str)))

for row in list(make2013.index):
    for column in list(make2013.columns):
        make2013_14.loc[str(row),str(column)] = make2013.loc[row,column]
        
for row in list(make2014.index):
    for column in list(make2014.columns):       
        make2014_14.loc[str(row),str(column)] = make2014.loc[row,column]
        
for i in range(len(make2014_14.index)):
    for j in range(len(make2014_14.columns)):
        if make2014_14.iloc[i,j] == '...':
            make2014_14.iloc[i,j] = 0
        
        
#writer = 'make2013_14.xlsx'
#make2013_14.to_excel(writer,'2013')
#make2014_14.to_excel(writer,'2014')