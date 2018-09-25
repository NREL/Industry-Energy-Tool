# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:49:30 2017

@author: cmcmilla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

file_2014 = "C:\\Users\\cmcmilla\\Downloads\\EIA\\MECS\\2014\\table3_2.xlsx"
# file_2010 = "C:\\Users\\cmcmilla\\Downloads\\EIA\\MECS\\MECS2010_Table3_2.xls"

econ_file_2014 = \
    "C:\\Users\\cmcmilla\\Downloads\\EIA\\MECS\\2014\\table6_3.xlsx"
# econ_file_2010 = \
#     "C:\\Users\\cmcmilla\\Downloads\\EIA\\MECS\\2010_UsebyEcon_Table6_3.xls"

county_file = "Y:\\6A20\\Public\\ICET\\Data for calculations\\" + \
    "Data foundation\\" + \
        "County_IndustryDataFoundation_2014_update_20170910-0116.csv"
        
fips_mecs_file = "Y:\\6A20\\Public\\ICET\\Data for calculations\\" + \
    "Data foundation\\" + \
        "US_FIPS_Codes.csv"
        


def import_mecs(mecs_file, year):
    """
    Import and format MECS Table 3.2 (Energy Consumption as a Fuel
    by Mfg Industry and Region (trillion Btu))
    """
    # Need to specifiy different header rows based on formatting changes
    # between 2010 and 2014 MECS

    naics_update = pd.read_excel(
        "Y:\\6A20\\Public\\ICET\\Data for calculations\\" +\
        "Data foundation\\2007_to_2012_NAICS.xls", sheetname=0, skiprows=[0, 1],
        names=['2007 NAICS Code', '2012 NAICS Code'], parse_cols=[0, 2]
        )

    naics_update = dict(naics_update.values)

    if year == 2014:
        h_row = 10

        cols_renamed = \
            {'Code(a)': 'NAICS', 'Electricity(b)': 'Net_electricity',
            'Fuel Oil': 'Residual_fuel_oil', 'Fuel Oil(c)': 'Diesel',
            'Gas(d)': 'Natural_gas', 'natural gasoline)(e)': 'LPG_NGL',
            'and Breeze': 'Coke_and_breeze', 'Other(f)': 'Other'
            }

    else:
        h_row = 9

        cols_renamed = \
            {'Code(a)': 'NAICS', 'Electricity(b)': 'Net_electricity',
            'Fuel Oil': 'Residual_fuel_oil', 'Fuel Oil(c)': 'Diesel',
            'Natural Gas(d)': 'Natural_gas', 'NGL(e)': 'LPG_NGL',
            'and Breeze': 'Coke_and_breeze', 'Other(f)': 'Other'
            }

    mecs = pd.read_excel(mecs_file, sheetname=0, header=h_row)

    mecs.rename(columns=cols_renamed, inplace=True)

    mecs.dropna(inplace=True, axis=0, how='all')

    mecs.loc[:, 'Region'] = \
        mecs[mecs.Total.apply(lambda x: len(str(x)) > 7)].Total

    mecs.Region.fillna(method='ffill', inplace=True)

    mecs.dropna(axis=0, inplace=True)

    mecs.loc[:, 'NAICS'] = mecs.NAICS.apply(lambda x: int(x))

    mecs.replace(to_replace={'*': None, 'Q': None, 'W': None}, inplace=True)

    # Need to convert 2007 NAICS used for MECS 2010 to 2012 NAICS
    if year == 2010:

        update_index = \
            mecs[mecs.NAICS.apply(lambda x: len(str(x)))==6].index
    
        mecs.loc[update_index, 'NAICS'] = \
            mecs.loc[update_index, 'NAICS'].map(lambda x: naics_update[x])

        mecs = mecs.groupby(['Region', 'NAICS'], as_index=False).sum()

    else:

        mecs.drop('Subsector and Industry', axis=1, inplace=True)

    mecs.set_index(['Region', 'NAICS'], drop=True, inplace=True)

    return mecs

mecs_2014 = import_mecs(file_2014, 2014)

def draw_pcolor_plots(df, county_nformatted, comp_type, save_dir):
    """
    Plot and save matplotlib pcolor plots of differences between MECS years
    or MECS and IET-estimated energy by Census Region.
    """

    plot_df = pd.DataFrame(df, copy=True)
    
    plot_df.replace({np.inf: 0, np.nan:0}, inplace=True)

    # Set all changes above 200% (including np.inf values) to 200% to help with
    # visualization of relative values.
    if comp_type == '%':
#        plot_df.replace(to_replace={np.inf: 2}, inplace=True)
#        plot_df[plot_df > 2] = 2
        cb_label = "Difference from 2014 MECS (1%)"
    
    else:
        cb_label = "Difference from 2014 MECS (TBtu)"

    for r in ['Total United States', 'Midwest Census Region', 
        'West Census Region','Northeast Census Region', 'South Census Region']:
            
        naics_ticks = plot_df.loc[r].index.values
  
        # Create masked array to address NaN values
        # mask2 is an array where county_perc_diff == np.nan & 
        # county_nformatted > 0
        mask2 = np.array(
            df.loc[r].isnull() & np.array(county_nformatted.loc[r] > 0)
            )
        m = np.ma.masked_where(
            np.isinf(df.loc[r].values) | mask2, plot_df.loc[r] * 100
            )
        # plot % change as matplot lib im figure
        fig, ax = plt.subplots()
        pc = ax.pcolor(m.T, cmap='gnuplot')
        ax.set_title(r)
        ax.set_xticklabels('')
        ax.tick_params('x', which='major', bottom='off')
        ax.set_yticks(np.arange(len(plot_df.columns)) + 0.5, minor=True)
        ax.set_yticklabels(plot_df.columns, minor=True, fontsize=4)
        ax.set_yticklabels('')
        ax.tick_params('y', which='major', left='off')
        ax.set_xticks(np.arange(len(naics_ticks)) + 0.5, minor=True)
        ax.set_xticklabels(
            naics_ticks, minor=True, rotation='vertical', fontsize=4
            )
        #ax.tick_params('both', which='minor', labelsize=2)
        plt.figure(num=1, figsize=(13,8), dpi=80)
        #ax.yaxis.set_ticks(np.arange(len(testi)) + 0.5, minor=True)
        #ax.yaxis.set(ticks=np.arange(len(testi)), labels=testi, minor=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="1.5%", pad=0.03)
        cb = plt.colorbar(pc, cax=cax)
        cb.set_label(cb_label, size=8)
        cb.ax.tick_params(labelsize=4)
        #ax.grid(True, which='minor')
        #plt.figure(num=1, figsize=(13, 8), dpi=200)
        fig.savefig(
            save_dir + comp_type + '_' + r + '.pdf', bbox_inches='tight'
            )


# Import 2014 county-level data and select only manufacturing naics
county_2014 = pd.read_csv(county_file, low_memory=False)

county_2014 = pd.DataFrame(
    county_2014[county_2014.subsector.apply(lambda x: x in [31, 32, 33])], 
        copy=True
        )

# Some MECS regions are missing
fips_mecs = pd.read_csv(fips_mecs_file, index_col='COUNTY_FIPS',
                        usecols=['COUNTY_FIPS', 'MECS_Region'])

county_2014.loc[
    county_2014[county_2014.MECS_Region.isnull()].index, 'MECS_Region'
    ] = county_2014[county_2014.MECS_Region.isnull()].fips_matching.apply(
        lambda x: fips_mecs.loc[x])

def align_naics(mecs_2014, county_2014):
    """
    Align county-level 6-digit NAICS codes to various n-digit NAICS codes used
    in 2014 MECS.
    Returns dataframe with energy sums matched to MECS n-digit NAICS codes
    by Census Region.
    """

    fuel_types = ['Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL',
                  'Natural_gas','Net_electricity', 'Other',
                  'Residual_fuel_oil', 'Total']

    mecs_naics = pd.DataFrame(
        {'naics': mecs_2014.loc['Total United States'].index.values,
        'Nn': np.nan}
        )

    mecs_naics.loc[:, 'Nn'] = mecs_naics.naics.apply(lambda x: len(str(x)))

    county_nformatted = pd.DataFrame()

    for n in range(3, 7):
        county_2014.loc[:, 'N' + str(n)] = \
            county_2014.naics.apply(lambda x: int(str(x)[0:n]))
        
        county_merged = pd.merge(
            county_2014, mecs_naics[mecs_naics.Nn == n], how='inner',
            left_on='N' + str(n), right_on='naics'
            )

        county_grouped = \
            county_merged.groupby(
                ['MECS_Region', 'N' + str(n)], as_index=False
                )[fuel_types].sum()

        county_nformatted = county_nformatted.append(
            county_grouped, ignore_index=True
            )

        county_nformatted = \
            county_nformatted.append(
                county_grouped.groupby(
                    'N' + str(n), as_index=False
                    )[fuel_types].sum(), ignore_index=True
                )

    county_nformatted.loc[:, 'NAICS'] = \
        county_nformatted[['N3', 'N4', 'N5', 'N6']].sum(axis=1)

    county_nformatted.loc[:, 'NAICS'] = county_nformatted.NAICS.astype(np.int)

    county_nformatted.drop(['N3', 'N4', 'N5', 'N6'], axis=1, inplace=True)

    county_nformatted.replace({'Midwest': 'Midwest Census Region', 
        'West': 'West Census Region', 'South': 'South Census Region',
        'Northeast': 'Northeast Census Region', np.nan: 'Total United States'},
        inplace=True)

    county_nformatted.rename(columns={'MECS_Region': 'Region'}, inplace=True)

    county_nformatted.set_index(['Region', 'NAICS'], inplace=True)

    return county_nformatted

county_nformatted = align_naics(mecs_2014, county_2014)

# Calculated absolute and relative differences between IET-estimated energy and
# MECS energy.
# NaN values represent instances where MECS values == *, W, H.
county_abs_diff = county_nformatted.subtract(mecs_2014)

county_perc_diff = county_abs_diff.divide(mecs_2014)

# Comparing at 6-digit NAICS level only
def n_NAICS_select(df, n):
    county_nD = pd.DataFrame(df.reset_index()[df.reset_index().NAICS.apply(
        lambda x: len(str(x)) == n
        )])

    county_nD.set_index(['Region', 'NAICS'], inplace=True, drop=True)
    
    return county_nD
    
county_abs_6D = n_NAICS_select(county_abs_diff, 6)

county_perc_6D = n_NAICS_select(county_perc_diff, 6)

def draw_econ_pcolor_plots(df, comp_type, save_dir):
    """
    Plot and save matplotlib pcolor plots of differences between MECS
    energy consumption ratios.
    """

    plot_df = pd.DataFrame(df, copy=True)

    # Set all changes above 200% (including np.inf values) to 200% to help with
    # visualization of relative values.
#    if comp_type == '%':
#        plot_df.replace(to_replace={np.inf: 2}, inplace=True)
#        plot_df[plot_df > 2] = 2    

    for r in ['Per_employee', 'Per_value_add', 'Per_value_shipments']:
            
        naics_ticks = plot_df[r].index.levels[0].values
        
        pivot_df = plot_df[r].reset_index().pivot(
            index='Economic_characteristic', columns='NAICS'
            )

        # Create masked array to address NaN values
        m = np.ma.masked_where(np.isnan(pivot_df[r].values), pivot_df[r])
        # plot % change as matplot lib im figure
        fig, ax = plt.subplots()
        pc = ax.pcolor(m)
        ax.set_title(r)
        ax.set_xticklabels('')
        ax.tick_params('x', which='major', bottom='off')
        ax.set_yticks(
            np.arange(len(plot_df[r].index.levels[1])) + 0.5, minor=True
            )
        ax.set_yticklabels(plot_df[r].index.levels[1], minor=True, fontsize=4)
        ax.set_yticklabels('')
        ax.tick_params('y', which='major', left='off')
        ax.set_xticks(np.arange(len(naics_ticks)) + 0.5, minor=True)
        ax.set_xticklabels(
            naics_ticks, minor=True, rotation='vertical', fontsize=4
            )
        #ax.tick_params('both', which='minor', labelsize=2)
        plt.figure(num=1, figsize=(13,8), dpi=80)
        #ax.yaxis.set_ticks(np.arange(len(testi)) + 0.5, minor=True)
        #ax.yaxis.set(ticks=np.arange(len(testi)), labels=testi, minor=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="1.5%", pad=0.03)
        cb = plt.colorbar(pc, cax=cax)
        cb.set_label(comp_type + ' Difference')
        cb.ax.tick_params(labelsize=4)
        #ax.grid(True, which='minor')
        #plt.figure(num=1, figsize=(13, 8), dpi=200)
        fig.savefig(
            save_dir + comp_type + '_' + r + '.pdf', bbox_inches='tight'
            )

    