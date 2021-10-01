#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:24:03 2021

@author: ewachs
"""


import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts, dim



####################### Read data files #####################

Use_Phys = "Physical_Use_Summary.xlsx"
Use_Mon = "Monetary_Use_Summary.xlsx"
BEA_Grouping = "BEA_Grouping.xlsx"

Use_Table_Physical = pd.read_excel(Use_Phys, sheet_name = None, header = 0, index_col = 0)
Use_Table_Monetary = pd.read_excel(Use_Mon, sheet_name = None, header = 0, index_col = 0)
BEA_Groups = pd.read_excel(BEA_Grouping, header = 0, index_col = None)

Years = list(range(2010, 2020, 1) )
Years_strings = list(map(str, Years))

###################### Methods ##############################

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

def tidy_dict(dict_df, typeIO = 'Use'):
    """
    Parameters
    ----------
    dict_dfs : dictionary of dataframes, here intended for import and export data.

    Returns
    -------
    dict_dfs : processed dictionaries.

    """
    tidy_dict = {}
    for key_dict, dataf in dict_df.items():
        #f_name = key_dict
        tidy_df = pd.melt(dataf.reset_index(), id_vars = 'index')
        Probs = tidy_df[tidy_df['value'] <= 0].index
        tidy_df = elim_nan(tidy_df)
        tidier_df = tidy_df.copy()
        tidier_df.drop(Probs, inplace = True)
        tidier_df.columns = ['source', 'target', 'value']
        if typeIO == 'Use':
            tidier_df.value = tidier_df.value * 1e-6
        else:
            tidier_df.value = tidier_df.value * 1e-3    
        tidy_dict[key_dict] = tidier_df
    return tidy_dict  

def reduce_dict(dict_df, list_keys):
    smaller_dict = {}
    for key, df in dict_df.items():
        if key in list_keys:
            smaller_dict[key] = dict_df[key]
            
    return smaller_dict

def make_dict_groups(df):
    dict_groups = {'group': '0', 'index': '0'}
    glist = []
    i = 0
    for rows in range(len(df)):
        dict_groups['group'] = df.iloc[i, 1]
        dict_groups['index'] = df.iloc[i, 0]
        glist.append(dict_groups)
        dict_groups = {'group': '0', 'index': '0'}
        i = i + 1
    
    return glist   

def make_chords(dict_df, type_IO = '_PhysUse'):
    hv.extension('bokeh')
    hv.output(size=250)
    for key_dict, dataf in dict_df.items():
        links = dict_df[key_dict]
        nodes = hv.Dataset(pd.DataFrame(node_d), 'index')
        chord = hv.Chord((links, nodes)).select(value=(5, None))
        chord.opts(
            opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                labels='name', node_color=dim('index').str()))
        hv.save(chord, 'chord_' + str(key_dict) + str(type_IO) + '.html')

        
        
################ Preprocessing #################################

Usable_Use_Mon = reduce_dict(Use_Table_Monetary, Years_strings)
link_Use_Mon = tidy_dict(Usable_Use_Mon, "Mon")
link_Use_Phys = tidy_dict(Use_Table_Physical)

hv.extension('bokeh')
hv.output(size=250)

node_d = make_dict_groups(BEA_Groups)

make_chords(link_Use_Phys)
make_chords(link_Use_Mon, '_MonUse')


