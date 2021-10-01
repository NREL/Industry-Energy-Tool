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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


################################ Import data #################################
filename = "US_Imports_HS.xlsx"
Imports_HS = pd.read_excel(filename, sheet_name=None)

filename = "US_Exports_HS.xlsx"
Exports_HS = pd.read_excel(filename, sheet_name=None)

# concordance data
ToBEA2012 = "BEA_NAICS_2012.xlsx"
HStoNAICS = "imp-code.xlsx"

# # data imports
BEA2012 = pd.read_excel(ToBEA2012, header = 4, usecols = "C, J")
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

# Preprocess Import Data by year
def clean_imp_exp_data(df):
    """
    Parameters
    ----------
    df : import or export data for a single year.

    Returns
    -------
    df : the import or export data for a single year after pre-processing.

    """
    df["ProductCode"] = df["ProductCode"].astype(str)
    df["ProductCode"] = df["ProductCode"].apply(fill_zeros)

    # Only want values with values in kg
    df = df.dropna(subset = ['NetWeight in KGM'])
    df = df[df['NetWeight in KGM'] > 0]
    
    return df

# Apply preprocessing to entire dictionary of dataframes
def apply_preprocessing(dict_dfs):
    """
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.

    Returns
    -------
    dict_dfs : processed dictionaries.

    """
    for key_dict, dataf in dict_dfs.items():
        f_name = key_dict
        dataf = clean_imp_exp_data(dataf) 
        dict_dfs[key_dict] = dataf
    return dict_dfs    


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

def Sim_conditional_merge(df1, df2, detail_level = 'Summary'):
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

    available = Imports_2012_BEA_from_HS[Imports_2012_BEA_from_HS[detail_level].isna()]
    Imports_2012_BEA_from_HS = Imports_2012_BEA_from_HS[Imports_2012_BEA_from_HS[detail_level].notna()]

    available = available.drop(columns = [detail_level, 'Related_2012_NAICS_Codes'])
    Imports_BEA_2_HS = pd.merge(available, df2, left_on = ['NAICS_5'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_2_HS[Imports_BEA_2_HS[detail_level].isna()]
    Imports_BEA_2_HS = Imports_BEA_2_HS[Imports_BEA_2_HS[detail_level].notna()]
    available = available.drop(columns = [detail_level, 'Related_2012_NAICS_Codes'])
    Imports_BEA_3_HS = pd.merge(available, df2, left_on = ['NAICS_4'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_3_HS[Imports_BEA_3_HS[detail_level].isna()]
    Imports_BEA_3_HS = Imports_BEA_3_HS[Imports_BEA_3_HS[detail_level].notna()]
    available = available.drop(columns = [detail_level, 'Related_2012_NAICS_Codes'])
    Imports_BEA_4_HS = pd.merge(available, df2, left_on = ['NAICS_3'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    available = Imports_BEA_4_HS[Imports_BEA_4_HS[detail_level].isna()]
    Imports_BEA_4_HS = Imports_BEA_4_HS[Imports_BEA_4_HS[detail_level].notna()]
    available = available.drop(columns = [detail_level, 'Related_2012_NAICS_Codes'])
    Imports_BEA_5_HS = pd.merge(available, df2, left_on = ['NAICS_2'], right_on = ['Related_2012_NAICS_Codes'], how = 'left')

    pdListHS = [Imports_2012_BEA_from_HS, Imports_BEA_2_HS, Imports_BEA_3_HS, Imports_BEA_4_HS, Imports_BEA_5_HS]
    return pdListHS


def apply_NAICS(dict_dfs, concordance_df):
    """
    
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.

    Returns
    -------
    new_dict : dictionary of dataframes for each year with unique NAICS codes included.

    """
    new_dict = {}
    #dict_df_NAICS
    for key_dict, dataf in dict_dfs.items():
        f_name = key_dict
        df_NAICS = pd.merge(dataf, concordance_df, left_on = ['ProductCode'], right_index = True, how = 'left')
        df_NAICS = df_NAICS[df_NAICS['NAICS'].notna()]
        df_NAICS = Make_NAICS_Cols(df_NAICS)
        new_dict[f_name] = df_NAICS
        
    return new_dict  

def apply_dupe_codes(dict_dfs, concordance_df):
    """
    
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.
    concordance_df : probably the HS_Dupe_Conc

    Returns
    -------
    new_dict : dictionary of dataframes for each year with unique NAICS codes included.

    """
    new_dict = {}
    #dict_df_NAICS
    for key_dict, dataf in dict_dfs.items():
        f_name = key_dict
        df_NAICS = pd.merge(dataf, HS_Dupe_Conc, left_on = ['ProductCode'], right_on = ['HS'])
        # Imports_BEA_dupes_HS = pd.merge(Imports_2012_NAICS_from_HS, HS_Dupe_Conc, left_on = ['Commodity Code'], right_on = ['HS'])
        new_dict[f_name] = df_NAICS
    return new_dict  

def NAICS_to_BEA(dict_df, Conc_df):
    for key_dict, dataf in dict_df.items():
        f_name = key_dict
        pdListHS = Sim_conditional_merge(dataf, Conc_df)
        BEA_HS = pd.concat(pdListHS)
        dict_df[key_dict] = BEA_HS
    return dict_df
        
def All_BEA(dict_df_unique, dict_df_dupes):
    new_dict = {}
    for key_dict, dataf in dict_df_unique.items():
        BEA_HS_df  = pd.concat([dataf, dict_df_dupes[key_dict]])
        new_dict[key_dict] = BEA_HS_df
    return new_dict

# ### Now that the imports have been assigned to BEA codes, compute price vectors for each BEA code
def Compute_price_vectors(dict_df):
    new_dict = {}
    for key_dict, dataf in dict_df.items():
        dataf.Summary = dataf.Summary.map(str)
        Price_Vector = dataf.groupby(by = ["Summary"]).sum()
        Price_Vector['Price'] = Price_Vector['TradeValue in 1000 USD']/Price_Vector['NetWeight in KGM']
        Price_Vector.index = Price_Vector.index.map(str)
        Price_Vector_fin = Price_Vector[['TradeValue in 1000 USD', 'NetWeight in KGM', 'Price']].copy()
        new_dict[key_dict] = Price_Vector_fin
    return new_dict    

def gather_prices(df):
    df_Info = df.copy()
    df_Price = pd.DataFrame(data = df_Info['TradeValue in 1000 USD']/df_Info['NetWeight in KGM'], index = df_Info.index, columns = ['Price'])
    df_Price['Summary'] = df_Info['Summary']
    df_Price = df_Price.sort_values(by = ['Summary'])
    return df_Price
    
    

def violin_plot_price(df1, df2, fig_name):


#    # Initialize the figure
#    f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = True, figsize = (12, 8))
    f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = True)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
    
#     # make and save the violin plots
    sns.violinplot(x = "Summary", y = "Price", data = df1, ax = ax1, scale = 'width', cut = 0)
    ax1.set(xlabel = "BEA Code", ylabel = "Import Price")
    sns.violinplot(x = "Summary", y = "Price", data = df2, ax = ax2, scale = 'width', cut = 0)
    ax2.set(xlabel = "BEA Code", ylabel = "Export Price")

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()    

def dict_to_xls(dict_df, file_name):
    """
    Write excel file from a dictionary of dataframes with each dataframe as sheet

    Parameters
    ----------
    dict_df : dictionary of dataframes to be recorded as sheets in excel workbook.
    file_name : filename to use to save workbook e.g. "example.xlsx"

    Returns
    -------
    None.

    """
 #   writer = pd.ExcelWriter(file_name, engine = 'xlsxwriter')
    with pd.ExcelWriter(file_name) as writer:
        for df_name, df in dict_df.items():
            df.to_excel(writer, sheet_name = df_name)
        



# ################################ Preprocessing ##################################
# # keep only imports    
# # remove repetitive data
# # apply zeros at the beginning of HS codes where Excel removed it   
# # only use data in kg

Imports_Data = apply_preprocessing(Imports_HS)
Exports_Data = apply_preprocessing(Exports_HS)


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
    Set = df1.Summary

    if len(Set) == 1:
        HS_Dupes = HS_Dupes.drop(items)
        BEA_Codes = list(Set)
        BEA_Conc_Index.append(items)
        BEA_Conc.append(BEA_Codes)

HS_Dupe_Conc = pd.DataFrame(np.column_stack([BEA_Conc_Index, BEA_Conc]), columns = ['HS', 'Summary'])        

#### Creates a dataframe with HS codes that are not multiply mapped

Imports_NAICS_from_HS = apply_NAICS(Imports_Data, HS_NAICS_Conc_Usable)
Exports_NAICS_from_HS = apply_NAICS(Exports_Data, HS_NAICS_Conc_Usable)

# ######## 

Imports_BEA_dupes = apply_dupe_codes(Imports_Data, HS_Dupe_Conc)
Exports_BEA_dupes = apply_dupe_codes(Exports_Data, HS_Dupe_Conc)


Imports_BEA = NAICS_to_BEA(Imports_NAICS_from_HS, BEA2012)
All_Imports_BEA = All_BEA(Imports_BEA, Imports_BEA_dupes)
Exports_BEA = NAICS_to_BEA(Exports_NAICS_from_HS, BEA2012)
All_Exports_BEA = All_BEA(Exports_BEA, Exports_BEA_dupes)

# ### Now that the imports have been assigned to BEA codes, compute price vectors for each BEA code


Import_Price_Vector = Compute_price_vectors(All_Imports_BEA)
Export_Price_Vector = Compute_price_vectors(All_Exports_BEA)


# see that 339 and 331 have prices much different than range, so exclude
All_Import_Prices = gather_prices(All_Imports_BEA['2019'])
All_Export_Prices = gather_prices(All_Exports_BEA['2019'])
Exclusions = ['113FF', '325', '326', '331', '332', '334', '339', 'Used']
All_Is = All_Import_Prices[~All_Import_Prices['Summary'].isin(Exclusions)]
All_Es = All_Export_Prices[~All_Export_Prices['Summary'].isin(Exclusions)]

Other_Is = All_Import_Prices[All_Import_Prices['Summary'].isin(Exclusions)]
Other_Es = All_Export_Prices[All_Export_Prices['Summary'].isin(Exclusions)]

Other_Exclusion = ['339']

Other_Is = Other_Is[~Other_Is['Summary'].isin(Other_Exclusion)]
Other_Es = Other_Es[~Other_Es['Summary'].isin(Other_Exclusion)]


Last_Is = All_Import_Prices[All_Import_Prices['Summary'].isin(Other_Exclusion)]
Last_Es = All_Export_Prices[All_Export_Prices['Summary'].isin(Other_Exclusion)]


# violin_plot_price(All_Is, All_Es, 'violins_2019_most_ps.svg')
# violin_plot_price(Other_Is, Other_Es, 'violins_2019_others.svg')
# violin_plot_price(Last_Is, Last_Es, 'violins_2019_339.svg')

dict_to_xls(Import_Price_Vector, 'Import_Price_Vector_Summary_Level.xlsx')
dict_to_xls(Export_Price_Vector, 'Export_Price_Vector_Summary_Level.xlsx')

# Percent difference of prices, just to know
Price_differences = np.zeros((25, 10))
i = 0
for key, df in Import_Price_Vector.items():
    Price_differences[:,i] = pct_diff(Import_Price_Vector[key].Price, Export_Price_Vector[key].Price)
    i = i+1
Years = list(range(2010, 2020, 1) )   
Price_diffs = pd.DataFrame(data = Price_differences, index = Import_Price_Vector['2010'].index, columns = Years)    
    

plt.figure(figsize = (10,6))
fmt = lambda x,pos: '{:.0%}'.format(x)
sns.heatmap(Price_diffs, yticklabels = True, cmap = 'YlGnBu', fmt = '.0%', cbar_kws = {'format': FuncFormatter(fmt)})
# fig, ax = plt.subplots()
# sns.heatmap(Price_diffs, yticklabels = True, cbar_kws = {'label': 'My Label'})
# #ax = sns.heatmap(Price_diffs, yticklabels = True, cbar = False)
# #cbar = ax.figure.colorbar('YlGnBu')
# #cbar.set_ticks([0, 1.6])
# #cbar.set_ticklabels(["0%", "160%"])
# cbar = ax.collections[1].colorbar
# cbar.ax.yaxis.set_major_formatter(PercentFormatter(1,0))

