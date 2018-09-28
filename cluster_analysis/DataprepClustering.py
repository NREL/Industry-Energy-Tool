# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 2017

@author: cmcmilla
"""

import pandas as pd
import numpy as np
import county_emps

###############################################################################
# Dataset directories
e_datadir = "Y:/6A20/Public/ICET/Data for calculations/Data foundation/"
eu_datadir = "Y:/6A20/Public/ICET/Data for calculations/Stock turnover/"

###############################################################################
# Data files
# NAICS definition file
naics_file = 'AEO_ind_NAICS.csv'

# County-level socio-economic data file
county_socio_file = 'county_socioecon.csv'

# County-level energy file
county_energy_file = \
    'County_IndustryDataFoundation_2014_update_20170910-0116.csv'

# End use files are compressed as .gz
mfg_enduse_file = 'manufacturing_EndUse.gz'
agri_enduse_file = 'agriculture_EndUse.gz'
mining_enduse_file = 'mining_EndUse.gz'

# FIPS codes
fips_file = 'US_FIPS_Codes.csv'

###############################################################################


def FIPS_dict():
    """
    Import FIPS codes and create a dictionary of FIPS: county, state
    """

    data = \
        dict(pd.read_csv(fips_file, usecols=['COUNTY_FIPS', 'Cty_St']).values)

    return data


def CountyEnergy_import():
    """
    Import and format county energy data
    """

    county_energy = \
        pd.read_csv(e_datadir + county_energy_file, low_memory=False)

    return county_energy


def CountySocio_import():
    """
    Import socio-economic data file as a dataframe.
    """
    county_socio = pd.read_csv(county_socio_file, index_col=['FIPS'])

    return county_socio


def cty_indemp_counts(n_NAICS, cty_indemp):
    """
    Create county industrial employment sums by specified n-NAICS
    to normalize energy data prior to cluster analysis.
    """
    
    # The counts method currently doesn't work
    # cty_indemp_counts = county_emps.cty_indemp.counts(n_NAICS)

    cty_indemp_counts = {}

    cty_indemp_counts['ag'] = \
        cty_indemp.cbp[cty_indemp.cbp.naics == 11].groupby(
            'fips_matching'
            ).emp.sum()

    cty_indemp_counts['non_ag'] = \
        cty_indemp.cbp[
            (cty_indemp.cbp.naics_n == n_NAICS) &
            (cty_indemp.cbp.subsector != 11)
            ].groupby(['fips_matching', 'naics']).emp.sum()

    return cty_indemp_counts


def CountyEU_import():
    """
    Import and format county-level energy by end use.
    """

    enduse = pd.DataFrame()

    for f in [mfg_enduse_file, agri_enduse_file, mining_enduse_file]:
        f_pd = pd.read_csv(eu_datadir + f, compression='gzip',
            index_col=[0])

        enduse = enduse.append(f_pd, ignore_index=True)

    # Drop end use categories that are sums of other categories.
    drop_euses = ['Indirect Uses-Boiler Fuel', 'Direct Uses-Total Process',
                  'Direct Uses-Total Nonprocess']

    enduse = pd.DataFrame(
        enduse[enduse.Enduse.apply(lambda x: x in drop_euses) == False],
        copy=True
        )

    enduse.replace(
        {'Enduse': {'Facility HVAC (g)': 'Facility HVAC'}}, inplace=True
        )

    enduse.loc[:, 'Total'] = enduse[
        ['Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL', 'Natural_gas',
        'Residual_fuel_oil', 'Net_electricity', 'Other']
         ].sum(axis=1)

    enduse['subsector'] = enduse.naics.apply(lambda x: int(str(x)[0:2]))

    return enduse


def naics_group_eu(enduse, county_ind_emp6, norm=False):
    """
    Frmat county-level energy by end use for cluster analysis. Option to
    normalize energy by six-digit NAICS employment.
    Returns a dictionary with input arrary and end uses arrary.
    """

    df = pd.DataFrame(enduse, copy=True)

    input_eu = {}

    if norm:

        tot_emp = \
            county_ind_emp6['non_ag'].reset_index().groupby(
                'fips_matching'
                ).emp.sum().add(county_ind_emp6['ag'], fill_value=0)

        df.drop(['subsector', 'MECS_NAICS', 'MECS_Region'], axis=1,
                inplace=True
                )

        df = df.groupby(['fips_matching', 'Enduse']).sum()

        df = pd.pivot_table(
            df.reset_index(), index='fips_matching', columns='Enduse',
            values='Total', aggfunc='sum'
            )

        df = pd.concat([df, tot_emp], axis=1, join='inner')

        df = df.divide(df.emp, axis=0)

        # Normalize by total number of industry employees.    
        df.drop('emp', axis=1, inplace=True)

        df.fillna(0, inplace=True)

        ctyfips = df.index.values

        input_eu['Enduse'] = df.columns.values

        input_eu['cla_array'] = df.values

    else:
        df = pd.DataFrame(
            df.groupby(
                ['fips_matching', 'Enduse'], as_index=False
                )['Total'].sum()
            )

        ctyfips = df.fips_matching.drop_duplicates()

        input_eu['Enduse'] = df.Enduse.drop_duplicates()

        input_eu['cla_array'] = df.pivot(
                index='fips_matching', columns='Enduse', values='Total'
                ).fillna(0).values

    return input_eu, ctyfips


def naics_group(county_energy, n_NAICS, county_ind_emp, norm=False):
    """
    Method to create an array for cluster analysis from a specified number of
    NAICS-code grouping of county-level energy data.
    """

    df = pd.DataFrame(county_energy, copy=True)

    cla_input = {}

    df.loc[:, 'naics'] = \
        df.naics.apply(lambda x: int(str(x)[0:n_NAICS]))

    if n_NAICS == 2:

        df.replace(
            {'naics': {32: 31, 33: 31}},
            inplace=True
            )

    if norm:

        # Treat agriculture data separately from others because USDA collects
        # employment data. Sum total energy by all ag NAICS and divide by
        # sum of all ag employment.
        df_ag = pd.DataFrame(df[df.subsector == 11], copy=True)

        df_ag.loc[:, 'naics'] = 11

        df = pd.DataFrame(df[df.subsector != 11], copy=True)

        for d in [df, df_ag]:
            d.drop(
                ['subsector', 'MECS_NAICS', 'MECS_Region', 'fipscty', 
                'fipstate'], axis=1, inplace=True
                )

        df = df.groupby(['fips_matching', 'naics']).sum()

        df_ag = df_ag.groupby(['fips_matching', 'naics']).sum()

        # Normalize by number of employees by NAICS
        df_ag = df_ag.divide(
            county_ind_emp['ag'], axis='index', level='fips_matching'
            )

        df = pd.concat([df, county_ind_emp['non_ag']], axis=1, join='inner')

        #df = df.groupby(['fips_matching', 'naics']).sum()

        df.loc[:, 'Total'] = df.Total.divide(df.emp, fill_value=0)

        df = pd.concat([df.reset_index(), df_ag.reset_index()], axis=0)

        df.fillna(0, inplace=True)

        df.reset_index(inplace=True, drop=False)

        # Drop inf values from final array input for cluster analysis
        ctyfips = df[df.Total != np.inf].fips_matching.drop_duplicates()
        cla_input['naics'] = df[df.Total != np.inf].naics.drop_duplicates()

        cla_input['cla_array'] = df[df.Total != np.inf].pivot(
            index='fips_matching', columns='naics',
            values='Total'
            ).fillna(0).values

    else:

        df.drop(
            ['subsector', 'MECS_NAICS', 'MECS_Region', 'fipscty', 
            'fipstate'], axis=1, inplace=True
            )

        df = df.groupby(['fips_matching', 'naics']).sum()

        df.reset_index(inplace=True, drop=False)

        ctyfips = df.fips_matching.drop_duplicates()

        cla_input['naics'] = df.naics.drop_duplicates()

        cla_input['cla_array'] = df.pivot(
            index='fips_matching', columns='naics', values='Total'
            ).fillna(0).values

    return cla_input, ctyfips
