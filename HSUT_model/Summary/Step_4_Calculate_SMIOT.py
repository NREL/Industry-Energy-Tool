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

# Price Vector derived in step 2
Import_Price_Vectors = "Use_Prices.xlsx"
Export_Price_Vectors = "Supply_Prices.xlsx"
Inverse_Use_Price_Matrix = 'Inverse_Price_Vectors_Use.xlsx'

Use_Mon = 'Monetary_Use_Summary.xlsx'
Make_Mon = 'Monetary_Make_Summary.xlsx'
q_Mon = 'Monetary_q.xlsx'
e_Mon = 'Monetary_e.xlsx'
x_Mon = 'Monetary_x.xlsx'

U_dict = pd.read_excel(Use_Mon, sheet_name = None, header = 0, index_col = 0)
V_dict = pd.read_excel(Make_Mon, sheet_name = None, header = 0, index_col = 0)
q_dict = pd.read_excel(q_Mon, sheet_name = None, header = 0, index_col = 0)
e_dict = pd.read_excel(e_Mon, sheet_name = None, header = 0, index_col = 0)
g_dict = pd.read_excel(x_Mon, sheet_name = None, header = 0, index_col = 0)

######################## Define methods ##########################################



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

# Apply preprocessing to entire dictionary of dataframes
# drop the first row of use 
# keep just the dfs corresponding to the years of interest
# keep only the relevant columns
def apply_preprocessing_use(dict_dfs):
    """
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.

    Returns
    -------
    dict_dfs : processed dictionaries.

    """
    for key_dict, dataf in dict_dfs.items():
        #f_name = key_dict
        dataf = format_idx_col(dataf) 
        dataf = elim_nan(dataf)
        dict_dfs[key_dict] = dataf
    return dict_dfs  


def invert_df(df):
    """
    Parameters
    ----------
    df : dataframe (square matrix)

    Returns
    -------
    df : inverse of original dataframe
    """
    df_inv = pd.DataFrame(np.linalg.pinv(df.values), df.columns, df.index)
    return df_inv

def make_diag_matrix(df, col):
    values = np.diag(df[col])
    df_hat = pd.DataFrame(values, index = df.index, columns = df.index)
    df_hat = format_idx_col(df_hat)
    return df_hat

def create_dict_inv_price_matrix(dict_dfs, column_to_use = 'Price'):
    new_dict = {}
    for key, df in dict_dfs.items():
        matrix_df = make_diag_matrix(df, column_to_use)
        inv_mat_df = invert_df(matrix_df)
        new_dict[key] = inv_mat_df
    return new_dict

def create_dict_diag_matrix(dict_dfs, column_to_use = 'Price'):
    new_dict = {}
    for key, df in dict_dfs.items():
        matrix_df = make_diag_matrix(df, column_to_use)
        new_dict[key] = matrix_df
    return new_dict   
  
def mult_df_dicts(dict_df1, dict_df2):
    Product_dict = {}
    for key, df in dict_df1.items():
        Prod = df.dot(dict_df2[key])
        Product_dict[key] = Prod
    return Product_dict

def transpose_dict(dict_df):
    transposed_dict = {}
    for key, df in dict_df.items():
        transposed_dict[key] = df.T
    return transposed_dict

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

def drop_rows_dict(dict_df, row_name_list):
    newdict = {}
    for key, df in dict_df.items():
        newdict[key] = df[~df.index.isin(row_name_list)]
    return newdict

def drop_columns_dict(dict_df, col_name_list):
    newdict = {}
    for key, df in dict_df.items():
        newdict[key] = df.drop(columns = col_name_list)
    return newdict

def Compute_A(U, V, g_diag_inv, q_diag_inv):
    Arg1 = U.dot(g_diag_inv)
    Arg2 = V.dot(q_diag_inv)
    A = Arg1.dot(Arg2)
    return A
    
            
################### Set up matrices and variables #########################


g_diag_inv_dict = create_dict_inv_price_matrix(g_dict, 'Total Industry Output')
q_diag_inv_dict = create_dict_inv_price_matrix(q_dict, 'Total Commodity Output')
q_diag_dict = create_dict_diag_matrix(q_dict, 'Total Commodity Output')
Arg1 = mult_df_dicts(U_dict, g_diag_inv_dict)
Arg2 = mult_df_dicts(V_dict, q_diag_inv_dict)
A_Mon_dict = mult_df_dicts(Arg1, Arg2)
Z_Mon_dict = mult_df_dicts(A_Mon_dict, q_diag_dict)

dict_to_xls(A_Mon_dict, 'A_Monetary_SIOT.xlsx')
dict_to_xls(Z_Mon_dict, 'Z_Monetary_SIOT.xlsx')
