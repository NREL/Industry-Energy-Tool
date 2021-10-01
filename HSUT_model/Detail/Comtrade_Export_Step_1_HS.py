#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:05:55 2021

@author: ewachs
"""
# File reads in Comtrade Import data
# Output is a price vector in 2012 dollars by BEA IO code

import pandas as pd
import numpy as np

################################ Import data #################################
#import data
Comtrade_HS_data = "US2012Exporttoworld_HS_Codes.xlsx"

# concordance data
ToBEA2012 = "BEA_NAICS_2012.xlsx"
HStoNAICS = "imp-code.xlsx"

# data imports
Exports_HS_2012 = pd.read_excel(Comtrade_HS_data, header = 0, usecols = "C:N")
BEA2012 = pd.read_excel(ToBEA2012, header = 4, usecols = "G, J")
HS_NAICS2012 = pd.read_excel(HStoNAICS, usecols = "A, H", dtype=str, header=None, names=['HS_10', 'NAICS'])

################## Methods #########################
# Fills first place in string with 0 value if relevant (HS 6 digit codes, Excel data was missing initial zeroes)
def fill_zeros(x):
    if len(x) < 6:
        return x.zfill(6)
    else:
        return x

# Computes percent difference
def pct_diff(x, y):
    z = abs(x - y)/(0.5*(x + y))
    return z

def Check_Multiple_Mappings(df, col_name = 'HS_6_dig'):
    """
    Check if 6 digit HS codes map to multiple NAICS codes

    Parameters
    ----------
    df : dataframe, HS to NAICS concordance
    col_name : string, name of column with NAICS codes. The default is 'HS_6_dig'.

    Returns
    -------
    DF_HS_Non_unique : dataframe with index that lists all HS codes, and values are number of 
    NAICS codes returned in mapping.

    """
    k = df['HS_6_dig'].unique()


    Count = []
    for codes in range(len(k)):
        df_mm = df.loc[df['HS_6_dig'] == k[codes]]
        df_mm = df_mm.reset_index()
        inner = pd.Series(data = df_mm.NAICS, dtype = 'str')
        Count.append(inner)

    Count2 = pd.DataFrame(Count).T
    Count2.columns = k

    Unique_NAICS = []

    for uvalues in k:
        Unique_Count = Count2.loc[:,uvalues].nunique()
        Unique_NAICS.append(Unique_Count)

    DF_HS_Non_unique = pd.DataFrame(Unique_NAICS).T
    DF_HS_Non_unique.columns = k
    DF_HS_Non_unique = DF_HS_Non_unique.T
    return DF_HS_Non_unique

def Make_NAICS_Cols(df):
    """
    Parameters
    ----------
    df : DataFrame with a column of 6 digit NAICS codes labeled 'NAICS'
        

    Returns
    -------
    df2 : DataFrame with NAICS codes broken down into 2, 3, 4, and 5 digits as well.

    """

    df1 = df['NAICS'].astype(str).str[0:2]
    df2 = df['NAICS'].astype(str).str[0:3]
    df3 = df['NAICS'].astype(str).str[0:4]
    df4 = df['NAICS'].astype(str).str[0:5]
    dfs = pd.DataFrame({'NAICS_2': df1, 'NAICS_3': df2, 'NAICS_4': df3, 'NAICS_5': df4})
    df2 = df.join(dfs)
    return df2

def Sim_conditional_merge(df1, df2):
    """
    Parameters
    ----------
    df1 : dataframe, has values for NAICS 2, 3, 4, 5 and 6 digit codes, named 'NAICS_#'.
    df2 : dataframe, the concordance between NAICS and BEA. In this function the 
    concordance column should be named 'Related_2012_NAICS_Codes' and the BEA column is 'Detail'

    Returns
    -------
    pdListHS : list of dataframes
    
    Since there is no conditional merge in pandas, the below code
# merges based on 6 digit NAICS code if available, then 5, 4, 3 or 2 digit codes

    """
    
    columns_wanted = ['NAICS', 'NAICS_2', 'NAICS_3', 'NAICS_4', 'NAICS_5']

    df1[columns_wanted] = df1[columns_wanted].astype(str)
    df2.Related_2012_NAICS_Codes = df2.Related_2012_NAICS_Codes.astype(str)

    Imports_2012_BEA_from_HS = pd.merge(df1, df2, left_on = ['NAICS'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_2012_BEA_from_HS.query('Detail.isnull()', engine = 'python')
    Imports_2012_BEA_from_HS = Imports_2012_BEA_from_HS[Imports_2012_BEA_from_HS['Detail'].notna()]

    available = available.drop(columns = ['Detail', 'Related_2012_NAICS_Codes'])
    Imports_BEA_2_HS = pd.merge(available, df2, left_on = ['NAICS_5'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_2_HS.query('Detail.isnull()', engine = 'python')
    Imports_BEA_2_HS = Imports_BEA_2_HS[Imports_BEA_2_HS['Detail'].notna()]
    available = available.drop(columns = ['Detail', 'Related_2012_NAICS_Codes'])
    Imports_BEA_3_HS = pd.merge(available, df2, left_on = ['NAICS_4'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_3_HS.query('Detail.isnull()', engine = 'python')
    Imports_BEA_3_HS = Imports_BEA_3_HS[Imports_BEA_3_HS['Detail'].notna()]
    available = available.drop(columns = ['Detail', 'Related_2012_NAICS_Codes'])
    Imports_BEA_4_HS = pd.merge(available, df2, left_on = ['NAICS_3'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_4_HS.query('Detail.isnull()', engine = 'python')
    Imports_BEA_4_HS = Imports_BEA_4_HS[Imports_BEA_4_HS['Detail'].notna()]
    available = available.drop(columns = ['Detail', 'Related_2012_NAICS_Codes'])
    Imports_BEA_5_HS = pd.merge(available, df2, left_on = ['NAICS_2'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    pdListHS = [Imports_2012_BEA_from_HS, Imports_BEA_2_HS, Imports_BEA_3_HS, Imports_BEA_4_HS, Imports_BEA_5_HS]
    return pdListHS



################################ Preprocessing ##################################
# keep only imports    
# remove repetititve data
# apply zeros at the beginning of HS codes where Excel removed it   
# only use data in kg

Exports_HS_2012["ProductCode"] = Exports_HS_2012["ProductCode"].astype(str)
Exports_HS_2012["ProductCode"] = Exports_HS_2012["ProductCode"].apply(fill_zeros)

# Only want values with values in kg
Exports_HS_2012 = Exports_HS_2012[Exports_HS_2012['NetWeight in KGM'] > 0]

# Published concordance of HS/NAICS is at 10 digit HS level, but import/export data available is at 6 digit level
HS_NAICS2012['HS_6_dig'] = HS_NAICS2012['HS_10'].str[0:6]

# NAICS codes in concordance use x rather than 0 to indicate comprehensive coverage of related codes, switch for compatibility with BEA
HS_NAICS2012.NAICS = HS_NAICS2012.NAICS.str.replace('X', '0')


########## Check if 6 digit HS codes can map to multiple NAICS codes ############

### Puts any HS 6 digit codes that map to multiple NAICS codes as index to dataframe whose value
# is the number of NAICS codes mapped

DF_HS_Non_unique = Check_Multiple_Mappings(HS_NAICS2012)


HS_Dupes = DF_HS_Non_unique[(DF_HS_Non_unique > 1).any(1)]
HS_Usable = DF_HS_Non_unique[(DF_HS_Non_unique == 1).any(1)]
HS_NAICS2012 = HS_NAICS2012.set_index('HS_6_dig')
HS_NAICS = HS_NAICS2012
HS_NAICS = HS_NAICS.drop(columns = 'HS_10')
HS_NAICS = HS_NAICS[~HS_NAICS.index.duplicated(keep = 'first')]


# Make a concordance only for HS values with unique mapping to NAICS codes
HS_NAICS_Conc_Usable = pd.merge(HS_Usable, HS_NAICS, left_index = True, right_index = True)


# The HS Dupes whose multiple NAICS codes all map to the same BEA code form a separate concordance
BEA_Conc_Index = []
BEA_Conc = []

for items in HS_Dupes.index:
    df = HS_NAICS2012.loc[HS_NAICS2012.index == items]
    df = df.NAICS.unique()
    df1 = BEA2012[BEA2012.Related_2012_NAICS_Codes.astype(str).isin(df)]
    Set = df1.Detail

    if len(Set) == 1:
        HS_Dupes = HS_Dupes.drop(items)
        BEA_Codes = list(Set)
        BEA_Conc_Index.append(items)
        BEA_Conc.append(BEA_Codes)

HS_Dupe_Conc = pd.DataFrame(np.column_stack([BEA_Conc_Index, BEA_Conc]), columns = ['HS', 'Detail'])        

#### Creates a dataframe with HS codes that are not multiply mapped
Exports_2012_NAICS_from_HS = pd.merge(Exports_HS_2012, HS_NAICS_Conc_Usable, left_on = ['ProductCode'], right_index = True, how = 'left')


######## 

Exports_2012_HS_EZ = Exports_2012_NAICS_from_HS[Exports_2012_NAICS_from_HS['NAICS'].notna()]
Exports_2012_HS_EZ = Make_NAICS_Cols(Exports_2012_HS_EZ) 

pdListHS = Sim_conditional_merge(Exports_2012_HS_EZ, BEA2012)
Exports_BEA_dupes_HS = pd.merge(Exports_2012_NAICS_from_HS, HS_Dupe_Conc, left_on = ['ProductCode'], right_on = ['HS'])

Exports_BEA_HS = pd.concat(pdListHS)
Exports_BEA_HS = pd.concat([Exports_BEA_HS, Exports_BEA_dupes_HS])

### Now that the imports have been assigned to BEA codes, compute price vectors for each BEA code
Exports_BEA_HS.Detail = Exports_BEA_HS.Detail.map(str)

HS_Price_Vector = Exports_BEA_HS.groupby(by = ["Detail"]).sum()
HS_Price_Vector['Price'] = 1000 * HS_Price_Vector['TradeValue in 1000 USD']/HS_Price_Vector['NetWeight in KGM']
HS_Price_Vector.index = HS_Price_Vector.index.map(str)


# # Store price vector in excel
writer = "Export_Price_Vector_HS.xlsx"
HS_Price_Vector.to_excel(writer, 'HS_Price_Vector')

