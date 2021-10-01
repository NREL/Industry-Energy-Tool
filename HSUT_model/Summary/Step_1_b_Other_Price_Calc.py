#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:05:55 2021

@author: ewachs
"""
# File reads in Comtrade Import data


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


################################ Import data #################################
filename = "US_Imports_HS.xlsx"
Imports_HS = pd.read_excel(filename, sheet_name=None)

filename = "US_Exports_HS.xlsx"
Exports_HS = pd.read_excel(filename, sheet_name=None)

######## Production Data ##############

Phys_Production = 'Production_Data.xlsx'
Production_Annual = pd.read_excel(Phys_Production, header = 0, usecols = 'A:M', index_col = 0, nrows = 21)

######## BEA Gross Output Data ##################
Gross_Output_Detail = 'Gross_Output_by_Industry_Detail_2010-2019.xls'
Output_Value = pd.read_excel(Gross_Output_Detail, header = 5, usecols = 'B:L', index_col = 0)

# concordance data
ToBEA2012 = "BEA_NAICS_2012.xlsx"
HStoNAICS = "imp-code.xlsx"

# # data imports
BEA2012 = pd.read_excel(ToBEA2012, header = 4, usecols = "C, G:H, J", nrows = 505)
HS_NAICS2012 = pd.read_excel(HStoNAICS, usecols = "A, H", dtype=str, header=None, names=['HS_10', 'NAICS'])


############# Time Series #######################

List_years = list(map(str, range(2010, 2020, 1)) ) 

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
        df_NAICS = pd.merge(dataf, concordance_df, left_on = ['ProductCode'], right_on = ['HS'])
        # Imports_BEA_dupes_HS = pd.merge(Imports_2012_NAICS_from_HS, HS_Dupe_Conc, left_on = ['Commodity Code'], right_on = ['HS'])
        new_dict[f_name] = df_NAICS
    return new_dict  

def NAICS_to_BEA(dict_df, Conc_df):
    for key_dict, dataf in dict_df.items():

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


# ################################ Preprocessing ##################################
Production_Annual = format_idx_col(Production_Annual)
Production_Annual['BEA Summary'] = Production_Annual['BEA Summary'].astype(str)
Production_Annual['BEA Detail'] = Production_Annual['BEA Detail'].astype(str)

Index_for_dfs = list(Production_Annual['BEA Detail'].unique())
Total_Prod_by_Detail = Production_Annual.groupby('BEA Detail').sum()
Gross_Output_w_Detail_Code = pd.merge(Output_Value, BEA2012, left_on = Output_Value.index, right_on = 'Industry Title')
Gross_Output_w_Detail_Code['Detail'] = Gross_Output_w_Detail_Code['Detail'].astype(str)
Gross_Output_w_Detail_Code = format_idx_col(Gross_Output_w_Detail_Code)


All_Output_Value = pd.merge(Gross_Output_w_Detail_Code, Total_Prod_by_Detail, left_on = 'Detail', right_on = 'BEA Detail', suffixes = ('bil_$', '_t'))
All_Output_Value_Summary = All_Output_Value.groupby('Summary').sum()
Alt_Prices = All_Output_Value[['Detail', 'Summary']]
#Alt_Prices_Summary = All_Output_Value[['Summary']]
for years in List_years:
    Alt_Prices[years] = All_Output_Value[years + 'bil_$'] * 1e6 / (1000 * All_Output_Value[years + '_t'])
    All_Output_Value_Summary[years + 'price thou$/kg'] = All_Output_Value_Summary[years + 'bil_$'] * 1e6 / (1000 * All_Output_Value_Summary[years + '_t'])
Alt_Prices_detail = Alt_Prices.drop_duplicates(keep = 'first')

Last_columns_size = len(List_years)

Alt_Prices_detail.to_excel('Alt_Prices_Detail_Level.xlsx')
Alt_Prices_Summary = All_Output_Value_Summary.iloc[:, list(range(-Last_columns_size, 0))]  
Alt_Prices_Summary.columns = List_years
Alt_Prices_Summary.to_excel('Alt_Prices_Summary_Level.xlsx')




