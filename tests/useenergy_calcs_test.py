# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 09:22:12 2017

@author: vnarwade
"""

import pandas as pd
import numpy as np

eio_input_output_file = 'USEEIO_InputOutput.xlsx'
input_output_file = 'IO_baseline.xlsx'

use_energy = pd.DataFrame(pd.read_excel('IO_baseline.xlsx', 
                                        sheetname = 'calcs_energy_use'))

input_output_dict = dict(pd.read_excel(input_output_file,
                                       sheetname = 'NAICS-IO',
                                       usecols = ['NAICS', 'BEA']).values)

eio_naics_dict = dict(pd.read_excel(eio_input_output_file, 
                                    sheetname = 'EIA Code-NAICS',
                                    usecols = ['EIA Code','NAICS']).values)

use_energy = use_energy.iloc[:13,:]

use_energy = use_energy.transpose()
naics = use_energy.iloc[:,2].map(eio_naics_dict)

io_code = pd.DataFrame(naics.map(input_output_dict))
io_code.columns = ['io_code']
use_energy = use_energy.join(io_code)
use_energy = use_energy.iloc[:,3:]
use_energy.fillna(0, inplace = True)
use_energy = use_energy.groupby(['io_code']).sum()

row = list(pd.read_excel(input_output_file, sheetname = 'use_energy_Quad').columns.astype(str))

use_energy_Quad = pd.DataFrame(index = row, columns = use_energy.columns)

for item in list(use_energy.index):
    for column in list(use_energy.columns):
        use_energy_Quad.loc[str(item), column] = use_energy.ix[item,column]
        
writer = 'use_energy_cals.xlsx'
use_energy_Quad.to_excel(writer, 'Sheet1')
