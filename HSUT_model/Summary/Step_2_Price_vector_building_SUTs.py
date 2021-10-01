#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:50:39 2021

@author: ewachs
"""
# File reads in Comtrade Import data
# Output is a price vector in 2012 dollars by BEA IO code

import pandas as pd
import numpy as np

################################ Import data #################################
#import data
# Price Vectors derived from HS codes
PV_IM = "Import_Price_Vector_Summary_Level.xlsx"
PV_EX = "Export_Price_Vector_Summary_Level.xlsx"

# Alt Price Vector
PV_Alt = "Alt_Prices_Summary_Level.xlsx"

# Use matrix after redefinitions (BEA summary, producer prices, 1997-2019)
IOUse = "IOUse_After_Redefinitions_PRO_1997-2019_Summary.xlsx"

Use_Monetary = pd.read_excel(IOUse, sheet_name = None, usecols = 'A, C:BU', header = 5, index_col = 0, nrows = 74)
Import_Price_vector = pd.read_excel(PV_IM, sheet_name = None, usecols = 'A, D', header = 0, index_col = 0)
Export_Price_vector = pd.read_excel(PV_EX, sheet_name = None, usecols = 'A, D', header = 0, index_col = 0)
Alt_Prices = pd.read_excel(PV_Alt, index_col = 0)

Years = list(range(2010, 2020, 1) ) 

######################## Define methods ##########################################
def make_set(Prefix_List, Big_Set):
    """
    
    Parameters
    ----------
    Prefix_List : list of prefixes in BEA 6-digit code that indicate the category, prefixes should all have the same length, at least one member 
    Big_Set : Dataframe of all BEA codes being used

    Returns
    -------
    Category_Set : Set consisting of all BEA codes in category

    """
    Prefix_Length = len(Prefix_List[0])
    Category = Big_Set[Big_Set.Code.str[0:Prefix_Length].isin(Prefix_List)]
    Category_Set = set(Category.Code)
    return Category_Set

def elim_nan(df):
    """
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe with nan values replaced with zero
        DESCRIPTION.
    """
    df = df.replace('\..+', np.nan, regex=True)
    df = df.fillna(np.nan)
    df = df.fillna(0)
    return df

def format_idx_col(df):
    """
    Parameters
    ----------
    df : dataframe.

    Returns
    -------
    dataframe with columns and index as string.
    """
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    return df

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
            
# Apply preprocessing to entire dictionary of dataframes
# drop the first row of use 
# keep just the dfs corresponding to the years of interest
# keep only the relevant columns
def apply_preprocessing(dict_dfs, USE = 1):
    """
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.

    Returns
    -------
    dict_dfs : processed dictionaries.

    """
    for key_dict, dataf in dict_dfs.items():

        dataf = format_idx_col(dataf) 
        dataf = elim_nan(dataf)
        
        if USE == 1:
            dataf = dataf.drop(['IOCode'], axis = 0)

        dict_dfs[key_dict] = dataf
    return dict_dfs  

def reindex_dict(dict_dfs, index_dict):
    for key, df in dict_dfs.items():
        Newdf = df.reindex(index_dict[key].index).fillna(0)
        dict_dfs[key] = Newdf
    return dict_dfs

def update_price_dicts(dict_dfs, df):
    for key, dframe in dict_dfs.items():
        dframe.columns = [key]
        dframe.update(df, join = 'left', overwrite = True)
        dframe.columns = ['Price']
#        dict_dfs[key] = dframe
    return dict_dfs    

################### Set up matrices and variables #########################

Alt_Prices = format_idx_col(Alt_Prices)

Use_Monetary = apply_preprocessing(Use_Monetary)
Import_Price_vector = apply_preprocessing(Import_Price_vector, 0)
Export_Price_vector = apply_preprocessing(Export_Price_vector, 0)

Import_Prices = reindex_dict(Import_Price_vector, Use_Monetary)
Export_Prices = reindex_dict(Export_Price_vector, Use_Monetary)

Import_Prices = update_price_dicts(Import_Prices, Alt_Prices)
Export_Prices = update_price_dicts(Export_Prices, Alt_Prices)
    
                    
# # Store price vector in excel
dict_to_xls(Import_Prices, "Use_Prices_hybrid.xlsx")
dict_to_xls(Export_Prices, "Supply_Prices_hybrid.xlsx")


