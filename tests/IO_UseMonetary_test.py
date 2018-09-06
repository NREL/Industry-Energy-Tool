# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:19:07 2017

@author: vnarwade
"""

""" Collapses 389 x 389 make tables to 78 x 78 matrix """

import pandas as pd
import numpy as np


eio_input_output_file = 'USEEIO_InputOutput.xlsx'
input_output_file = 'IO_baseline.xlsx'

eio_table = pd.read_excel(eio_input_output_file, sheetname = 'US_2013_use')
#eio_table.columns = [list(eio_table.iloc[0,:])]
eio_naics_dict = dict(pd.read_excel(eio_input_output_file, 
                                    sheetname = 'EIA Code-NAICS',
                                    usecols = ['EIA Code','NAICS']).values)

naics_code = pd.DataFrame(eio_table['Commodity / Industry'].map(eio_naics_dict))
naics_code.rename(columns = {'Commodity / Industry':'naics'}, inplace = True)

eio_table = eio_table.join(naics_code)

input_output_dict = dict(pd.read_excel(input_output_file,
                                       sheetname = 'NAICS-IO',
                                       usecols = ['NAICS', 'BEA']).values)

IO_code = pd.DataFrame(eio_table.naics.map(input_output_dict))
IO_code.rename(columns = {'naics':'IO_code'}, inplace = True)
#IO_code.fillna(0,inplace = True)

eio_table = eio_table.join(IO_code)

eio_table.set_index(['IO_code'], inplace = True)



naics_row = eio_table.iloc[0,:].map(eio_naics_dict)
IO_code_row = pd.DataFrame(naics_row.map(input_output_dict))
IO_code_row.columns = ['IO_code']
eio_table = eio_table.append(IO_code_row.transpose())


eio_table = eio_table.transpose()

eio_table.dropna(axis = 0, subset = ['IO_code'], inplace = True)
eio_table.set_index(['IO_code'], inplace = True)

eio_table = eio_table.transpose()
eio_table = eio_table.reset_index().dropna(axis = 0, subset = ['index']).set_index('index')

eio_table.fillna(0, inplace = True)

eio_use_monetary = eio_table.reset_index().groupby(['index']).sum()
eio_use_monetary = eio_use_monetary.transpose().reset_index()
eio_use_monetary = eio_use_monetary.groupby(['IO_code']).sum().transpose()


        


#use_writer = 'use_monetary_2014.xlsx'
#eio_use_monetary.to_excel(use_writer, '2013')