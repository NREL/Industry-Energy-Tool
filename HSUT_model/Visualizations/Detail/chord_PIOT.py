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


PIOT = 'PIOTalt.xlsx'

PIOT_Z = pd.read_excel(PIOT, header = 0, index_col = 0)

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

tidy_PIOT = pd.melt(PIOT_Z.reset_index(), id_vars = 'index')
Probs = tidy_PIOT[tidy_PIOT['value'] < 0].index
tidy_PIOT = elim_nan(tidy_PIOT)
tidier_PIOT = tidy_PIOT.copy()
tidier_PIOT.drop(Probs, inplace = True)
Probs = tidy_PIOT[tidy_PIOT['value'] == 0].index
tidier_PIOT.drop(Probs, inplace = True)
tidier_PIOT.columns = ['source', 'target', 'value']
tidier_PIOT['value'] = 0.001 * tidier_PIOT['value']
hv.extension('bokeh')
hv.output(size=250)

links = tidier_PIOT

PIOT_Z['index1'] = PIOT_Z.index
PIOT_Z['group'] = PIOT_Z.index1.astype(str).str[0:2]
df = PIOT_Z[['index1', 'group']].copy().reset_index().drop('index', 1)
Key = pd.DataFrame(df.group.unique()).reset_index()
Key.columns = ['group', 'NAICS_2']
df_fin = df.merge(Key, left_on = ['group'], right_on = ['NAICS_2'])
df_fin = df_fin.drop(['group_x', 'NAICS_2'], 1)
df_fin.columns = ['index', 'group']
node_desc = df_fin.to_dict('records')

nodes = hv.Dataset(pd.DataFrame(node_desc), 'index')

chord = hv.Chord((links, nodes)).select(value=(5, None))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                labels='name', node_color=dim('index').str()))

hv.save(chord, 'chord_piotalt.html')
###### Do the same thing for the MIOT ##############

# MIOT = "SIOT_BEA2012AR_CxC.xlsx"

# Z_Monetary = pd.read_excel(MIOT, usecols = 'A: OP', skipfooter = 6, header = 0, index_col = 0)
# Z_Monetary = format_idx_col(Z_Monetary)

# tidy_MIOT = pd.melt(Z_Monetary.reset_index(), id_vars = 'index')
# Probs = tidy_MIOT[tidy_MIOT['value'] < 0].index
# tidy_MIOT = elim_nan(tidy_MIOT)
# tidier_MIOT = tidy_MIOT.copy()
# tidier_MIOT.drop(Probs, inplace = True)
# Probs = tidy_MIOT[tidy_MIOT['value'] == 0].index
# tidier_MIOT.drop(Probs, inplace = True)
# tidier_MIOT.columns = ['source', 'target', 'value']
# tidier_MIOT['value'] = 0.001 * tidier_MIOT['value']
# hv.extension('bokeh')
# hv.output(size=250)

# links = tidier_MIOT

# Z_Monetary['index1'] = Z_Monetary.index
# Z_Monetary['group'] = Z_Monetary.index1.astype(str).str[0:2]
# df = Z_Monetary[['index1', 'group']].copy().reset_index().drop('index', 1)
# Key = pd.DataFrame(df.group.unique()).reset_index()
# Key.columns = ['group', 'NAICS_2']
# df_fin = df.merge(Key, left_on = ['group'], right_on = ['NAICS_2'])
# df_fin = df_fin.drop(['group_x', 'NAICS_2'], 1)
# df_fin.columns = ['index', 'group']
# node_desc = df_fin.to_dict('records')

# nodes = hv.Dataset(pd.DataFrame(node_desc), 'index')

# chord = hv.Chord((links, nodes)).select(value=(5, None))
# chord.opts(
#     opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
#                 labels='name', node_color=dim('index').str()))

# hv.save(chord, 'chord_miot.html')