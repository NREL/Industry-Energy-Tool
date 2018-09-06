# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 13:42:24 2017

@author: vnarwade
"""

import pandas as pd
import numpy as np


eio_input_output_file = 'USEEIO_InputOutput.xlsx'
input_output_file = 'IO_baseline.xlsx'

eio_table = pd.read_excel(eio_input_output_file, sheetname = 'US_2013_use')
eio_table.columns = [list(eio_table.iloc[0,:])]
eio_naics_dict = dict(pd.read_excel(eio_input_output_file, 
                                    sheetname = 'EIA Code-NAICS',
                                    usecols = ['EIA Code','NAICS']).values)

naics_code = pd.DataFrame(eio_table.Code.map(eio_naics_dict))
naics_code.rename(columns = {'Code':'naics'}, inplace = True)

eio_table = eio_table.join(naics_code)

input_output_dict = dict(pd.read_excel(input_output_file,
                                       sheetname = 'NAICS-IO',
                                       usecols = ['NAICS', 'BEA']).values)

IO_code = pd.DataFrame(eio_table.naics.map(input_output_dict))
IO_code.rename(columns = {'naics':'IO_code'}, inplace = True)
#IO_code.fillna(0,inplace = True)

eio_table = eio_table.join(IO_code)
na = eio_table.iloc[0,:].map(eio_naics_dict)
IO_code_rows = pd.DataFrame(na.map(input_output_dict))
IO_code_rows.columns = ['IO_code']


#eio_table = eio_table.iloc[1:,2:]

#eio_table =eio_table.transpose().reset_index()
#eio_table = eio_table.join(IO_code_rows)
#eio
#naics = pd.DataFrame(eio_table[0].map(eio_naics_dict))
#naics.columns = ['naics']
#eio_table = eio_table.join(naics)
#IO = pd.DataFrame(eio_table.naics.map(input_output_dict))
#IO.columns = ['IO']
#eio_table = eio_table.join(IO)
#eio_table.drop(['index',0,'naics'], axis = 1, inplace = True)
##eio_table = eio_table.groupby(['IO']).sum().transpose()


#nan_index = eio_table[eio_table.IO_code == 0]
#nan_index.set_index(['naics','Unnamed: 1'], inplace = True)

#writer = 'Use.xlsx'
#eio_table.to_excel(writer, 'EIO-Use_monetary')


#modified_eio_input_output_file = 'EIA Use.xlsx'
#
#eio_use_monetary = pd.read_excel(modified_eio_input_output_file,
#                                  sheetname = 'EIO-Use_monetary')
#
#eio_use_monetary = eio_use_monetary.drop(['Code', 'Commodity Description', 'naics'], 
#                                             axis = 1)
#
#
#eio_use_monetary = eio_use_monetary.groupby(['IO_code']).sum()
#
#
#eio_use_monetary = eio_use_monetary.transpose()
#eio_use_monetary = eio_use_monetary.reset_index()
#eio_use_monetary['index'] = eio_use_monetary['index'].astype(str)
#eio_use_monetary['index'] = [this.split(".")[0] for this in eio_use_monetary['index']]
#eio_use_monetary = eio_use_monetary.groupby('index').sum().transpose()
#
#
#        
#
#
##use_writer = 'use_monetary_final.xlsx'
##eio_use_monetary.to_excel(use_writer, 'use_monetary')