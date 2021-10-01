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
PV_HS = "Estimated_Price_Vector.xlsx"
# Alt Price Vector
PV_Alt = "Alt_Prices_Detail_Level.xlsx"
# Total Requirements matrix (BEA det, commodity by commodity, 2012)
CC_DR = "SIOT_BEA2012AR_CxC.xlsx"
Use = 'IOUse_After_Redefinitions_PRO_DET.xlsx'
Make = 'IOMake_After_Redefinitions_DET.xlsx'


Z_Monetary = pd.read_excel(CC_DR, usecols = 'A: OP', skipfooter = 6, header = 0, index_col = 0)
Use_Table = pd.read_excel(Use, sheet_name = '2012', header = 5, index_col = 0)
Total_Final_Use = pd.read_excel(Use, sheet_name = '2012', header = 5, index_col = 0, usecols=('A, PN'))
Total_Final_Demand_Commodities = pd.read_excel(Use, sheet_name = '2012', header = 5, index_col = 0, usecols=('A, PM'))
Y = pd.read_excel(Use, sheet_name = '2012', usecols = 'A, PM', header = 5, index_col = 0)
Make_Table = pd.read_excel(Make, sheet_name = '2012', usecols = 'A, C:PN', header = 5, index_col = 0)
Price_vector = pd.read_excel(PV_HS, header = 0, index_col = 0)
Alt_Prices = pd.read_excel(PV_Alt, index_col = 0)
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
    df = df.fillna(np.nan)
    df = df.fillna(0)
    return df

def invert_df(df):
    """
    Parameters
    ----------
    df : dataframe

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

################### Set up matrices and variables #########################
# L_Monetary = format_idx_col(L_Monetary)
# L_Monetary = L_Monetary[:-1]   
#A_Monetary = format_idx_col(A_Monetary)
Z_Monetary = format_idx_col(Z_Monetary)  
Use_Monetary = format_idx_col(Use_Table)
Make_Monetary = format_idx_col(Make_Table)  
e_Monetary = format_idx_col(Total_Final_Demand_Commodities)
q_Monetary = format_idx_col(Total_Final_Use)
# A_Mon_vals = A_Monetary.drop(['Commodity Description'], axis = 1) 
X_vals = Make_Table.loc['T007', :].T
X_vals = X_vals.drop('T008')
X_vals = elim_nan(X_vals)
X = X_vals.to_frame()
X = format_idx_col(X)
X = X[:-2]
# X = X.reindex(Z_Monetary.index)
x_hat = pd.DataFrame(np.diag(X.T007), index = X.index, columns = X.index)
x_hat = format_idx_col(x_hat)
x_hat_inverse = invert_df(x_hat)

Y = Y.iloc[:-8]
Y = elim_nan(Y)
Y = format_idx_col(Y)

A_Monetary = Z_Monetary.dot(x_hat_inverse)
Price_vector = format_idx_col(Price_vector)
Alt_Prices = format_idx_col(Alt_Prices)
Alt_Prices_2012 = pd.DataFrame(Alt_Prices['2012'] * 1000)
Alt_Prices_2012.index = Alt_Prices['Detail']
Alt_Prices_2012.columns = ['Price']
Alt_Prices_2012.rename_axis('Index', inplace = True)

Prices = Price_vector.reindex(A_Monetary.index).fillna(0)
Prices.update(Alt_Prices_2012)

# # make diagonal matrix of prices

p_hat = pd.DataFrame(np.diag(Prices.Price), index = Prices.index, columns = Prices.index)
p_hat_inverse = invert_df(p_hat)
inverse_price_array = np.array(p_hat_inverse)
Eye = pd.DataFrame(np.eye(len(A_Monetary)), index = Prices.index, columns = Prices.index)


############################## Calculations ##############################


A_Monetary_transpose = A_Monetary.transpose()
Z_Monetary_transpose = Z_Monetary.transpose()
Z_phys_T = Z_Monetary_transpose.dot(p_hat_inverse)
Z_phys_T_test1 = A_Monetary.dot(x_hat)
Z_phys_T_test1 = Z_phys_T_test1.transpose()
Z_phys_T_test = Z_phys_T_test1.dot(p_hat_inverse)
Z_phys = Z_phys_T.transpose()


Y_phys = pd.DataFrame(p_hat_inverse.values.dot(Y.values), index = p_hat_inverse.index, columns = ['Y'])

X_phys = pd.DataFrame(p_hat_inverse.values.dot(X.values), index = p_hat_inverse.index, columns = ['X'])

X_phys_hat = make_diag_matrix(X_phys, 'X')
X_phys_hat_inverse = invert_df(X_phys_hat)
A_phys = Z_phys.dot(X_phys_hat_inverse)


# Calculate Use and Supply in physical terms
Use_Mon = elim_nan(Use_Monetary)
Make_Mon = elim_nan(Make_Monetary)
q_Mon = elim_nan(q_Monetary)
q_Mon = q_Mon[:-8]
e_Mon = elim_nan(e_Monetary)
e_Mon = e_Mon[:-8]

Use_Mon = Use_Mon.iloc[:, :-29]
Use_Mon = Use_Mon[:-8] #drops last 8 rows, which are not part of physical transactions portion of use table
Use_Mon = Use_Mon.drop(['Commodity Description'], axis = 1)
Make_Mon = Make_Mon.iloc[:, :-3]
Make_Mon = Make_Mon[:-1] #drops last 6 rows, which are not part of physical transactions portion of make table
Supply_Mon = Make_Mon.T
                       
Use_phys = Use_Mon.dot(inverse_price_array)
q_Phys = q_Mon.T.dot(p_hat_inverse)
e_Phys = e_Mon.T.dot(p_hat_inverse)
Use_phys.columns = Use_Mon.columns

Supply_phys = Supply_Mon.dot(inverse_price_array)
Supply_phys.columns = Supply_Mon.columns

# Store price vector in excel
writer = "PIOTalt.xlsx"
Z_phys.to_excel(writer, 'Z_phys') 
writer = "PUse2012alt.xlsx"
Use_phys.to_excel(writer, 'Use_phys')
writer = "PSupply2012alt.xlsx"
Supply_phys.to_excel(writer, 'Supply_phys')

