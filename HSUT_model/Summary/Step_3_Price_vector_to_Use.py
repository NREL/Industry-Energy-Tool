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
Import_Price_Vectors = "Use_Prices_hybrid.xlsx"
Export_Price_Vectors = "Supply_Prices_hybrid.xlsx"

Use = 'IOUse_After_Redefinitions_PRO_1997-2019_Summary.xlsx'
Make = 'IOMake_After_Redefinitions_1997-2019_Summary.xlsx'


Use_Table = pd.read_excel(Use, sheet_name = None, header = 5, index_col = 0)

Make_Table = pd.read_excel(Make, sheet_name = None, header = 5, index_col = 0)
Final_Demand = pd.read_excel(Use, sheet_name = None, header = 5, index_col = 0, usecols = 'A, BY:CR')
Total_Comm_Output_q = pd.read_excel(Use, sheet_name = None, header = 6, index_col = 0, usecols = 'A, CV')
Total_Final_Demand_e = pd.read_excel(Use, sheet_name = None, header = 6, index_col = 0, usecols = 'A, CU')
Total_Ind_Output_x = pd.read_excel(Make, sheet_name = None, header = 6, index_col = 0, usecols = 'A, BX')
Price_vector_Use = pd.read_excel(Import_Price_Vectors, sheet_name = None, header = 0, index_col = 0)
Price_vector_Make = pd.read_excel(Export_Price_Vectors, sheet_name = None, header = 0, index_col = 0)
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
def apply_preprocessing_use(dict_dfs, Use_Ind = 1):
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
        if Use_Ind == 1:
            dataf = dataf.iloc[:, :-27]
            dataf = dataf[:-14] #drops last 14 rows, which are not part of physical transactions portion of use table
            dataf = dataf.drop(['Commodities/Industries'], axis = 1)
            dataf = dataf.drop(['IOCode'], axis = 0)
        elif Use_Ind == 0:
            dataf = dataf.iloc[:, :-1]
            dataf = dataf.drop(['Industries/Commodities'], axis = 1)
            dataf = dataf[:-6] #drops last 6 rows, which are not part of physical transactions portion of make table
            dataf = dataf.drop(['IOCode'], axis = 0)
        elif Use_Ind == 2:
            dataf = dataf[:-14] #drops last 14 rows, which are not part of physical transactions portion of use table
        elif Use_Ind == 3:
            dataf = dataf[:-6] #drops last 14 rows, which are not part of physical transactions portion of use table    
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

def create_dict_inv_price_matrix(dict_dfs, price_column = 'Price'):
    for key, df in dict_dfs.items():
        matrix_df = make_diag_matrix(df, price_column)
        inv_mat_df = invert_df(matrix_df)
        dict_dfs[key] = inv_mat_df
    return dict_dfs
    
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
            
################### Set up matrices and variables #########################
Use_Monetary = apply_preprocessing_use(Use_Table)
Make_Monetary = apply_preprocessing_use(Make_Table, 0)
Final_Demand_Monetary = apply_preprocessing_use(Final_Demand, 2)
Total_Comm_Output_q_Monetary = apply_preprocessing_use(Total_Comm_Output_q, 2)
Total_Final_Demand_e_Monetary = apply_preprocessing_use(Total_Final_Demand_e, 2)
Total_Ind_Output_x_Monetary = apply_preprocessing_use(Total_Ind_Output_x, 3)

Inverse_Prices_Use = create_dict_inv_price_matrix(Price_vector_Use)
Inverse_Prices_Make = create_dict_inv_price_matrix(Price_vector_Make)

Commodities_Only = ['Used', 'Other']
Inverse_Prices_Make_Industry = drop_rows_dict(Inverse_Prices_Make, Commodities_Only)
Inverse_Prices_Make_Industry = drop_columns_dict(Inverse_Prices_Make_Industry, Commodities_Only)

Supply_Monetary = transpose_dict(Make_Monetary)


Use_Physical = mult_df_dicts(Inverse_Prices_Use, Use_Monetary)
Supply_Physical = mult_df_dicts(Inverse_Prices_Make, Supply_Monetary)
Make_Physical = transpose_dict(Supply_Physical)
q_Physical = mult_df_dicts(Inverse_Prices_Use, Total_Comm_Output_q_Monetary)
e_Physical = mult_df_dicts(Inverse_Prices_Use, Total_Final_Demand_e_Monetary)
x_Physical = mult_df_dicts(Inverse_Prices_Make_Industry, Total_Ind_Output_x_Monetary)

dict_to_xls(Total_Comm_Output_q_Monetary, 'Monetary_q.xlsx')
dict_to_xls(Total_Final_Demand_e_Monetary, 'Monetary_e.xlsx')
dict_to_xls(Total_Ind_Output_x_Monetary, 'Monetary_x.xlsx')
dict_to_xls(Inverse_Prices_Use, 'Inverse_Price_Vectors_Use.xlsx')
dict_to_xls(Use_Physical, "Physical_Use_Summary.xlsx")
dict_to_xls(Use_Monetary, "Monetary_Use_Summary.xlsx")
dict_to_xls(Make_Physical, "Physical_Make_Summary.xlsx")
dict_to_xls(Make_Monetary, "Monetary_Make_Summary.xlsx")
dict_to_xls(Supply_Physical, "Physical_Supply_Summary.xlsx")
dict_to_xls(q_Physical, "Physical_q.xlsx")
dict_to_xls(e_Physical, "Physical_e.xlsx")
dict_to_xls(x_Physical, "Physical_x.xlsx")


