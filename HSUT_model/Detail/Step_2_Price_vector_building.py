#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:50:39 2021

@author: ewachs
"""
# File reads in Comtrade Import data
# Output is a price vector in 2012 dollars by BEA IO code

import pandas as pd


################################ Import data #################################
#import data
# Price Vectors derived from HS codes
PV_HS = "Price_Vector_HS.xlsx"
PV_EX = "Export_Price_Vector_HS.xlsx"

# Total Requirements matrix (BEA det, commodity by commodity, 2012)
CC_TR = "CxC_TR_2007_2012_PRO_DET.xlsx"

L_Monetary = pd.read_excel(CC_TR, sheet_name = '2012', usecols = 'A, C:PM', header = 4, index_col = 0)
Import_Price_vector = pd.read_excel(PV_HS, usecols = 'A, H', header = 0, index_col = 0)
Export_Price_vector = pd.read_excel(PV_EX, usecols = 'A, H', header = 0, index_col = 0)


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

################### Set up matrices and variables #########################
L_Monetary = format_idx_col(L_Monetary)
L_Monetary = L_Monetary[:-1]   
Import_Price_vector = format_idx_col(Import_Price_vector)
Export_Price_vector = format_idx_col(Export_Price_vector)
Import_Prices = Import_Price_vector.reindex(L_Monetary.index).fillna(0)
Export_Prices = Export_Price_vector.reindex(L_Monetary.index).fillna(0)

Prices = Import_Prices
Prices = Prices.where(Prices != 0, other = Export_Prices)


######### Make sets for categories ####################
Big_Set = pd.DataFrame(data = Prices.index)

# make list of prefixes that correspond to known categories
Service_Prefixes = ['4', '5', '6', '7', '8', 'G', 'S']
Hidden_Services = ['115000', '21311A', '323120']
Construction_Prefixes = ['23']
Drilling_Prefixes = ['213111']
Utilities_Prefixes = ['22']

Services = make_set(Service_Prefixes, Big_Set)
Services2 = make_set(Hidden_Services, Big_Set)
Drilling = make_set(Drilling_Prefixes, Big_Set)
Utilities = make_set(Utilities_Prefixes, Big_Set)
Services = set.union(Services, Services2, Drilling, Utilities)

Construction = make_set(Construction_Prefixes, Big_Set)

# Find the zero price entries. Check if fall into services or construction. For others,
# check for 4-digit NAICS correspondence and if that is not available, for 3-digit NAICS correspondence
# keep data around approximated prices in dataframe
Zero_Prices = Prices[Prices.Price == 0]
To_Approximate = Zero_Prices[~Zero_Prices.index.isin(Services)]
To_Approximate = To_Approximate[~To_Approximate.index.isin(Construction)]

# Set up lists to store average prices, whether at 3 or four digit level, and min and max of those used
Avg_Like_Prices, Four_digit_avg, Three_digit_avg, Min_price, Max_price, Outliers = ([] for i in range(6))

for NAICS in To_Approximate.index:
    Prefix = NAICS[0:5]
    Prefix = [Prefix]
    Like = make_set(Prefix, Big_Set)
    Sim_Prices = Prices[Prices.index.isin(Like)]
    Similar_Prices = Sim_Prices[Sim_Prices.Price > 0]
    if len(Similar_Prices) > 0:
        Four_digit_avg.append(1)
        Three_digit_avg.append(0)
        Avg_Like_Prices.append(Similar_Prices['Price'].mean())
        Min_price.append(Similar_Prices['Price'].min())
        Max_price.append(Similar_Prices['Price'].max())
    else:
        Prefix = NAICS[0:4]
        Prefix = [Prefix]
        Like = make_set(Prefix, Big_Set)
        Sim_Prices = Prices[Prices.index.isin(Like)]
        Similar_Prices = Sim_Prices[Sim_Prices.Price > 0]
        if len(Similar_Prices) > 0:
                Three_digit_avg.append(1)
                Four_digit_avg.append(0)
                Avg_Like_Prices.append(Similar_Prices['Price'].mean())
                Min_price.append(Similar_Prices['Price'].min())
                Max_price.append(Similar_Prices['Price'].max())
        else:
                Outliers.append(NAICS)
                Three_digit_avg.append(0)
                Four_digit_avg.append(0)
                Avg_Like_Prices.append(0)
                Min_price.append(0)
                Max_price.append(0)
                
                
To_Approximate.Price = Avg_Like_Prices  
Prices.update(To_Approximate)              
                    
# # Store price vector in excel
writer = "Estimated_Price_Vector.xlsx"
Prices.to_excel(writer, 'Prices')    
