# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:34:47 2017

@author: vnarwade
"""


import pandas as pd
import numpy as np

county_file = 'County_IndustryDataFoundation_2014_update_20170515-1600.csv'

mecs_naics_file = '2-6 digit_2017_Codes.csv'

input_output_file = 'IO_baseline.xlsx'

    #Import county energy (in TBtu)
county_energy = pd.read_csv(
        county_file, index_col = ['fips_matching', 'naics'], 
                                 low_memory = False)

    #Need to remap 'MECS_NAICS' back to original NAICS used in MECS.
mecs_naics_dict = dict(pd.read_csv(
        mecs_naics_file,usecols = ['MECS_NAICS_dummies', 'MECS_NAICS']).values)

county_energy.MECS_NAICS = county_energy.MECS_NAICS.map(mecs_naics_dict)

county_energy.MECS_NAICS.fillna(0, inplace = True)

county_energy.MECS_NAICS = county_energy.MECS_NAICS.apply(lambda x: int(x))

county_energy.rename(columns = 
                     {'fips_matching.1': 'fips_matching', 'naics.1': 'naics'},
                     inplace = True)

input_output_dict = dict(pd.read_excel(input_output_file,
                                       sheetname = 'NAICS-IO',
                                       usecols = ['NAICS', 'BEA']).values)

IO_code = pd.DataFrame(county_energy.naics.map(input_output_dict))
IO_code.rename(columns = {'naics' : 'IO_code'}, inplace = True)
county_energy = county_energy.join(IO_code)


io_energy = county_energy.groupby(['IO_code']).sum()
io_energy.drop([
'fipscty','fips_matching', 'fipstate','naics','MECS_NAICS', 'subsector'],
 axis = 1, inplace = True)


