# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:19:03 2017

@author: vnarwade; updated by cmcmilla
"""

from __future__ import division
import proc_temp as proc_temp
import fuel_switching
import pandas as pd
import numpy as np
import itertools as itools
import copy


class Stock_Calcs(object):

    # Initialize with projection years, calculated by providing an ending year
    # (y_end) and a step.
    def __init__(self, y_end, step):

        self.y_end = y_end

        self.step = step

        self.proj_years = [2013, 2014]

        [self.proj_years.append(int(x)) for x in np.linspace(
            2015, self.y_end, num=(self.y_end - 2015)/self.step + 1,
            endpoint=True
            )]

        proj_years = self.proj_years

        self.fs_params = {}

        self.ee_params = {}

    def set_fs_params(self, change_2050, ind, eu, temp_band):

        self.fs_params['change_2050'] = change_2050

        self.fs_params['ind'] = ind

        self.fs_params['eu'] = eu

        self.fs_params['temp_band'] = temp_band

        fs_params = self.fs_params

    def set_ee_params(self, scaling, bandwidth):

        self.ee_params['scaling'] = scaling

        self.ee_params['bandwidth'] = bandwidth

        ee_params = self.ee_params

    ee_params = {}

    fs_params = {}

    datadir = 'Y:\\6A20\\Public\\ICET\\Data for calculations\\Stock turnover\\'

    stock_turnover_shipments_file = 'AEO_regional_VS.csv'

    county_energy_file = \
        'Y:\\6A20\\Public\\ICET\\Data for calculations\\Data foundation\\' + \
        'County_IndustryDataFoundation_2014_update_20170910-0116.csv'

    fips_region_file = 'census_region.csv'
    tpc_file = 'NEMS_IDM_TPCs_all_sectors.xlsx'
    naics_file = 'AEO_ind_NAICS.csv'
    mfg_enduse_file = 'manufacturing_EndUse.gz'
    agri_enduse_file = 'agriculture_EndUse.gz'
    mining_enduse_file = 'mining_EndUse.gz'

    fuel_types = ['Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL',
                  'Natural_gas', 'Net_electricity', 'Other',
                  'Residual_fuel_oil', 'Total_energy_use']

    @classmethod
    def VS(cls):
        """
        Import industry value of shipments projections from AEO. 
        Returns a dictionary of dataframes that separate value of shipments
        regional and national projections, and regional growth rates.
        """

        VS = {}

        VS['regional'] = \
            pd.read_csv(cls.datadir + cls.stock_turnover_shipments_file)

        regional_col_dict = {}

        for c in VS['regional'].columns[1:-2]:
            regional_col_dict[c] = int(c)

        VS['regional'].rename(columns=regional_col_dict, inplace=True)

        VS['regional'].rename(columns={'subsector': 'AEO_industry'}, inplace=True)

        VS['national'] = \
            pd.DataFrame(VS['regional'].drop(['region'], axis=1), copy=True)

        VS['national'] =VS['national'].groupby(['AEO_industry']).sum()

        VS['regional'].set_index(['region', 'AEO_industry'], inplace=True)

        VS['growth'] = pd.DataFrame(
            VS['regional'].reset_index().iloc[:, 4:].divide(
                VS['regional'].reset_index()[2014], axis='index'
                )
            )

        VS['growth'] = pd.concat(
            [VS['growth'],
             VS['regional'].reset_index()[['region', 'AEO_industry']]], axis=1
            )

        VS['growth'].set_index(['region', 'AEO_industry'], inplace=True)

        return VS

    @classmethod
    def Def_Dicts(cls):
        """
        Define a dictionary of dictionaries to be used elsehwere in
        calculations.
        """

        calc_dicts = {}

        calc_dicts['fips_census'] =  \
            dict(pd.read_csv(cls.datadir + cls.fips_region_file,
                             usecols=['FIPS', 'census_region']).values)

        calc_dicts['naics'] = \
            dict(pd.read_csv(cls.datadir + cls.naics_file,
                             usecols=['naics', 'stock_turnover_industry']
                             ).values)

        calc_dicts['enduse'] = \
            dict(pd.read_excel(cls.datadir + cls.tpc_file,
                               sheetname='enduse_ST',
                               use_cols=['end_use', 'stock_turnover_enduse']
                               ).values)

        calc_dicts['TPC_enduse'] = dict(
            pd.read_excel(cls.datadir + cls.tpc_file,
                          sheetname='enduse_TPC_ST',
                          use_cols=['TPC_enduse', 'stock_turnover_enduse']
                          ).values
            )

        return calc_dicts

    @classmethod
    def TPC_import(cls, calc_dicts):
        """
        Import and format NEMS assumptions for industrial technology
        possibility curves (TPCS)
        """
        tpc = {}
        
        tpc['new'] = \
            pd.read_excel(cls.datadir + cls.tpc_file, sheetname='UEC-New')

        tpc['existing'] = \
            pd.read_excel(cls.datadir + cls.tpc_file, sheetname='UEC-Existing')

        # Map end uses for TPCs to stock turnover end uses (ST_enduse)
        for df in tpc.keys():

            tpc[df].loc[:, 'ST_enduse'] = \
                tpc[df].Enduse.map(calc_dicts['TPC_enduse'])

            tpc[df].rename(columns={'industry': 'AEO_industry'}, inplace=True)
            
            # Calculating the mean of TPCs.
            tpc[df] = \
                tpc[df].groupby(
                    ['AEO_industry', 'ST_enduse']
                    ).mean().reset_index()

        return tpc

    @staticmethod
    def calc_stock_turnover(regional_VS, old_lifetime=20, new_lifetime=30,
                            linear=True):
        """
        Method for calculating stock turnover by aggregated industry and region.
        Currently only calculates linear decay of stock.
        Returns a dataframe of post 2015 base-year stock additions, 
        """

        # Defualt lifetime values for existing (old) and new stock
        if old_lifetime is None:

            old_lifetime = 20

        if new_lifetime is None:

            new_lifetime = 30

        regional_VS = regional_VS.reset_index().sort_values(
        	['region', 'AEO_industry'], axis=0
        	).set_index(['region', 'AEO_industry'])

        post_2014_additions = pd.DataFrame(columns=[regional_VS.columns],
                                                index=[regional_VS.index],
                                                copy=True)

        pre_2014_stock = pd.DataFrame(columns=[regional_VS.columns],
                                      index=[regional_VS.index], copy=True)


        for df in [post_2014_additions, pre_2014_stock]:

            df.fillna(0, inplace=True)

            df.index = pd.MultiIndex.from_tuples(
                df.index,names=['region', 'AEO_industry']
                )
        
        pre_2014_stock.iloc[:, 0:4] = regional_VS.iloc[:, 0:4]

        #pre 2014 stock decays linearly with a lifetime of 20 years
        pre_2014_retirements = pd.DataFrame(
            np.multiply(regional_VS[2014].values.reshape(120, 1),
                        np.repeat(1 / old_lifetime, old_lifetime)),
            columns=range(2015, 2015 + old_lifetime),
            index=regional_VS.index)

        pre_2014_retirements = pre_2014_retirements.loc[:, range(2015, 2051)]

        pre_2014_stock.loc[:, 4:] = \
            np.subtract(pre_2014_stock[2014].values.reshape(120, 1),
                        pre_2014_retirements.cumsum(axis=1))

        pre_2014_stock.fillna(0, inplace=True)

        # Post 2014 stock additions are the sum of the annual 
        # non-negative change in value of shipments and the 
        # replacement for pre_2014 stock retirements
        post_2014_additions.iloc[:, 4:] = \
            np.subtract(regional_VS.iloc[:, 4:].values,
                        regional_VS.iloc[:, 3:-1].values)

        post_2014_additions = \
            post_2014_additions.where(post_2014_additions > 0, 0)

        post_2014_additions.iloc[:, 4: 4 + old_lifetime] = \
            np.add(post_2014_additions.iloc[:, 4:4 + old_lifetime],
                   pre_2014_retirements)

        post_2014_stock = post_2014_additions.cumsum(axis=1)

        # Adjust pre_2014_stock and post_2014 stock to account for years where 
        # decreases in value of shipments occur so that 
        # pre_2014_stock + post_2014 stock = value of shipments

        adjustment = \
            np.where(np.subtract(
                regional_VS.iloc[:, 4:].values,
                regional_VS.iloc[:, 3:-1].values
                ) < 0,
                np.subtract(regional_VS.iloc[:, 4:].values,
                            regional_VS.iloc[:, 3:-1].values),
                0)

        pre_2014_stock.iloc[:, 4:4 + old_lifetime]= \
            np.add(pre_2014_stock.iloc[:, 4:4 + old_lifetime],
                   adjustment[:, 0:old_lifetime])

        post_2014_stock.iloc[:, 4 + old_lifetime:] = \
            np.add(post_2014_stock.iloc[:, 4 + old_lifetime:],
                   adjustment[:, old_lifetime:])

        post_2014_retirements = pd.DataFrame(
            np.multiply(post_2014_stock.values,
                        np.repeat(1 / new_lifetime,
                                  len(post_2014_stock.columns))),
            columns=post_2014_stock.columns,
            index=post_2014_stock.index)

        # First retirements of post-2014 stock occur in 2016 (stock in 2014)
        # is considered existing stock.
        post_2014_retirements.loc[:, 2015] = 0

        # Post-2014 additions to stock are the sum of value of shipment changes
        # and post-2014 retirements
        post_2014_additions = post_2014_additions.add(post_2014_retirements)

        stock_dict = {}

        stock_dict['new'] = post_2014_stock

        stock_dict['old'] = pre_2014_stock

        stock_dict['new_additions'] = post_2014_additions

        stock_dict['total'] = pre_2014_stock.add(post_2014_stock)

        # Drop AEO industries that are not used in calculations (
        # e.g., we currently have no end use information or the industry
        # has been disaggregated)            
        for df in stock_dict.keys():
            
            stock_dict[df].sort_index(inplace=True)

            stock_dict[df].drop(['Bulk Chemicals', 'Food', 'Construction'],
                                    axis=0, level=1, inplace=True)

        return stock_dict



    @classmethod
    def Calc_Enduse(cls, calc_dicts, tpc_new):
        """
        Method for importing and formatting county-level sector end use files
        and for aggregating to regional level. Returns dataframes for county-
        level end use and regional end use.
        Also adds temperature buckets for the manfuacturing sector.
        """

        cty_enduse = pd.DataFrame()

        for f in [cls.mfg_enduse_file, cls.agri_enduse_file,
                  cls.mining_enduse_file]:

            f = cls.datadir + f

            sector_enduse = pd.read_csv(f, compression='gzip', index_col=0)

            if 'manufacturing' in f:

                sector_enduse, temps = \
                    proc_temp.TempChar().ImportTemps(sector_enduse)

                sector_enduse = \
                    proc_temp.TempChar().heat_mapping(temps, sector_enduse,
                                                     char=None)
            else:

                pass

            sector_enduse.loc[:, 'subsector'] = \
                sector_enduse.naics.apply(lambda x: int(str(x)[0:2]))

            sector_enduse.loc[:, 'region'] = \
                sector_enduse.fips_matching.map(calc_dicts['fips_census'])

            sector_enduse.loc[:, 'AEO_industry'] = \
                sector_enduse.naics.map(calc_dicts['naics'])

            sector_enduse.loc[:, 'ST_enduse'] = \
                sector_enduse.Enduse.map(calc_dicts['enduse'])

            # Drop all aggregate end uses (e.g., indirect uses-boiler, process 
            # uses)
            sector_enduse.dropna(subset=['ST_enduse'], inplace=True)

            cty_enduse = pd.concat([cty_enduse, sector_enduse], axis=0,
                                   ignore_index=True)

        cty_grpd = cty_enduse.groupby(['AEO_industry', 'ST_enduse'],
                                      as_index=False)       

        for g in cty_grpd.groups:

            grp_index = cty_grpd.get_group(g).index

            val = \
                g not in tpc_new.groupby(['AEO_industry', 'ST_enduse']).groups

            cty_enduse.loc[grp_index, 'ALT_enduse'] = val

        # Aggregate to regional enduse
        reg_eu_cols = ['region', 'AEO_industry', 'naics', 'ST_enduse',
                       'ALT_enduse','100-249', '250-399', '400-999', '<100',
                       '>1000']

        for ft in cls.fuel_types:

            reg_eu_cols.append(ft)

        regional_enduse = pd.DataFrame(
            cty_enduse[reg_eu_cols].groupby(
                ['region', 'AEO_industry', 'naics', 'ST_enduse', 'ALT_enduse'],
                as_index=False
                ).sum()
            )

        # TPCs are not defined for all industries and end uses. Identify those
        # that require an alternate definition.

        regional_enduse.set_index(['region', 'AEO_industry'], inplace=True)

        return cty_enduse, regional_enduse


    @classmethod
    def calc_energy_proj(cls, proj_years, enduse, vs_growth,
                         geo='regional'):
        """
        Project energy use by fuel type and geography (region or county) based
        on NEMS value of industrial shipments projections. 
        Returns values in MMBtu
        """

        #Create copy of end use dataframe
        enduse_df = enduse.copy(deep=True)

        vs_growth_years = pd.DataFrame(vs_growth[proj_years], copy=True)

        vs_growth_years.sort_index(inplace=True)

        if enduse_df.index.names == ['region', 'AEO_industry']:

            pass

        else:

            if type(enduse_df.index) == pd.indexes.multi.MultiIndex:

                enduse_df.reset_index(inplace=True)

                enduse_df.set_index(['region', 'AEO_industry'],
                                    inplace=True)

            if type(enduse_df.index) == pd.indexes.range.RangeIndex:

                enduse_df.set_index(['region', 'AEO_industry'],
                                    inplace=True)

        enduse_df.sort_index(inplace=True)

        energy_columns = ['<100', '100-249', '250-399', '400-999', '>1000']

        [energy_columns.append(ft) for ft in cls.fuel_types]

        energy_proj = pd.DataFrame()

        if geo == 'regional':

            ep_index = enduse_df.index

            energy_array = \
                np.array(
                    [np.array([enduse_df[col] for col in energy_columns])]
                    )

            add_columns = ['naics', 'ST_enduse', 'ALT_enduse']

        else:

            # Drop aggregate end use categories (indirect uses, total process
            # uses, etc.)
            enduse_df.dropna(subset=['ST_enduse'], inplace=True)

            ep_index = enduse_df.index

            energy_array = \
                np.array(
                    [np.array(
                        [enduse_df[col] for col in energy_columns]
                        )]
                    )

            add_columns = ['naics', 'ST_enduse', 'ALT_enduse', 'fips_matching']

        vs_proj_array = vs_growth_years.reindex(index=ep_index.values).values

        energy_proj_array = np.multiply(
            np.stack(np.array(energy_array), axis=2), vs_proj_array
            ) * 1e6

        for col in energy_columns:

            ep_cols = list(itools.product([col], proj_years))

            energy_proj = \
                pd.concat([energy_proj, pd.DataFrame(
                    energy_proj_array[energy_columns.index(col)],
                    columns=pd.MultiIndex.from_tuples(ep_cols,
                                                      names=('temp_fuel',
                                                             'year'))
                    )], axis=1)

        energy_proj.set_index(ep_index, inplace=True)

        energy_proj = pd.concat([energy_proj, enduse_df[add_columns]],
                                axis=1, join='inner')

        energy_proj.set_index(add_columns, append=True, inplace=True)

        energy_proj.set_axis(1, pd.MultiIndex.from_tuples(
            energy_proj.columns, names=('temp_fuel', 'year')
            ))

        return energy_proj


    @classmethod
    def Calc_Baseline_uec(cls, reg_proj, county_enduse, tpc,
                          uec_type, detail=False):
        """
        Method for building unit energy consumption data for old stock,
        new stock and weighted average stock.
        """

        proj_years = reg_proj.columns.levels[1]

        if detail == False:

            df = pd.DataFrame(index=reg_proj.reset_index().set_index(
                    ['AEO_industry', 'ST_enduse'], drop=True
                    ).index.drop_duplicates())

        else:

            df = pd.DataFrame(
                county_enduse.reset_index()[['AEO_industry', 'ST_enduse',
                                             'region',
                                             'bw_naics']].drop_duplicates()
                )

            df_grpd = df.groupby(['AEO_industry', 'ST_enduse'])

            for g in df_grpd.groups:

                if np.nan in g:

                    continue

                else:

                    df.loc[df_grpd.get_group(g).index, 'ALT_enduse'] = \
                        reg_proj.reset_index(level=4, drop=False).xs(
                            g, level=[1, 3])[['ALT_enduse']].values[0]

            df.set_index(['AEO_industry', 'ST_enduse'], inplace=True)

            df.drop(['ALT_enduse'], axis=1, inplace=True)

        if uec_type in ['new', 'old']:

            if uec_type == 'new':
                
                tpc_type = tpc[uec_type]

            else:
                tpc_type = tpc['existing']

            df = pd.merge(df.reset_index(), tpc_type, how='left',
                            on=['AEO_industry', 'ST_enduse'])

            if detail == False:
                df = \
                    df.set_index(['AEO_industry', 'ST_enduse'])[proj_years[1:]]

            else:
                df = \
                    df.set_index(['AEO_industry', 'ST_enduse', 'region',
                                    'bw_naics'])[proj_years[1:]]

                df.reset_index(['region', 'bw_naics'], inplace=True)

            # Set all CHP and/or Cogeneration TPCs equal to Conventional Boiler
            # TPCs and all other missing end uses to equivalent 
            # Balance of Manufacturing end use.

            if detail == True:

                df_grpd = \
                    df.reset_index().groupby(['AEO_industry', 'ST_enduse'])

                df.reset_index(inplace=True)

                for g in df_grpd.groups:

                    missing_ind = g[0] not in tpc_type.AEO_industry.values

                    df_grpd_index = df_grpd.get_group(g).index

                    if (g[1] == 'CHP and/or Cogeneration') & \
                        (missing_ind == False):

                        df.loc[df_grpd_index, proj_years[1:]] = \
                            df_grpd.get_group(
                                (g[0],'Conventional Boiler')
                                ).iloc[0, 4:].values

                        # u_add = u_add.append(tpc_type.groupby(
                        #     ['AEO_industry', 'ST_enduse']
                        #     ).get_group((g[0], 'Conventional Boiler')))

                    if (g[1] == 'CHP and/or Cogeneration') & \
                        (missing_ind == True):

                        df.loc[df_grpd_index, proj_years[1:]] = \
                            df_grpd.get_group(
                                ('Balance of Manufacturing',
                                 'Conventional Boiler')
                                ).iloc[0, 4:].values

                    if g[1] != 'CHP and/or Cogeneration':

                        df.loc[df_grpd_index, proj_years[1:]] = \
                            df_grpd.get_group(('Balance of Manufacturing',
                                                g[1])).iloc[0, 4:].values

                df.set_index(['AEO_industry', 'ST_enduse', 'bw_naics'],
                            inplace=True)

            if detail == False:

                for i in df[df.loc[:, proj_years[1]].isnull()].index:

                    if (i[1] == 'CHP and/or Cogeneration') & \
                        (False in df.ix[
                            (i[0], 'Conventional Boiler')
                            ].isnull().values):

                        df.loc[i, :] = df.ix[(i[0], 'Conventional Boiler')]

                    if (i[1] == 'CHP and/or Cogeneration') & \
                        (True in df.ix[
                            (i[0], 'Conventional Boiler')
                            ].isnull().values):

                        df.loc[i, :] = df.ix[('Balance of Manufacturing',
                                            'Conventional Boiler')]

                    if i[1] != 'CHP and/or Cogeneration':

                        df.loc[i, :] = df.ix[('Balance of Manufacturing',
                                            i[1])]

        return df

    @classmethod
    def Define_Stock_Char(cls, county_enduse, energy_projections,
                          stock_dict, uecs, VS, bw_master_dict,
                          bw_input=None, fs_input=None):
        """
        Stock characteristics defined only on AEO Industry and Census Region
        levels due to limited resolution of value of shipments projections.
        """
        # Create copy data frames of baseline uecs.
        uecs_og = copy.deepcopy(uecs)

        years = energy_projections.columns.levels[1]

        initial_stock = copy.deepcopy(stock_dict)

        # Here new stock and old stock are in energy values
        new_stock = {}

        old_stock = {}
        
        energy_projections.sort_index(inplace=True)

        for col in energy_projections.columns.levels[0]:

            new_stock[col] = \
                energy_projections.reset_index().groupby(
                    ['region', 'AEO_industry', 'ST_enduse']
                    ).sum()[col][years]

            old_stock[col] = \
                energy_projections.reset_index().groupby(
                    ['region', 'AEO_industry', 'ST_enduse']
                    ).sum()[col][years]

        for k in initial_stock.keys():
            initial_stock[k] = initial_stock[k].reindex(
                index=new_stock['Coal'].index,
                method='ffill'
                )

        # Adjust uecs based on bandwidth efficiency input. Assumes both
        # old and new stock are adjusted by the same value.
        if bw_input is None:

            pass
        
        else:
    
            bw_change = pd.DataFrame(index=bw_input.index,
                                     columns=range(2014, years[-1]+1))

            for i in bw_change.index:

                bw_change.loc[i, :] = \
                    np.linspace(1, bw_input.ix[i].values,
                                len(range(years[0],
                                          years[-1])))

            bw_change = bw_change[years[1:]]

            for age in ['new', 'old']:
                
                uecs_og[age].reset_index(inplace=True)

                uecs_og[age].loc[:, 'NAICS'] = \
                    uecs_og[age].bw_naics.map(bw_master_dict)

                bw_uec_grp = uecs_og[age].groupby('NAICS')

                for g in bw_uec_grp.groups:

                    uecs_og[age].loc[bw_uec_grp.get_group(g).index,
                                  tuple(bw_change.columns)] = \
                                  uecs_og[age].loc[
                                      bw_uec_grp.get_group(g).index,
                                      tuple(bw_change.columns)
                                      ].multiply(bw_change.ix[g].values)
                                      
                uecs_og[age].drop(['NAICS'], axis=1, inplace=True)
                
                uecs_og[age].set_index(['AEO_industry', 'ST_enduse',
                                        'bw_naics'], inplace=True, drop=True)

        # Remove detail from uecs, so that data are indexed only to region,
        # AEO_industry, and ST_enduse.
        stock_uecs = {}
        
        if type(uecs_og['new'].index) == pd.indexes.multi.MultiIndex:

            for age in ['new', 'old']:
    
                stock_uecs[age] = uecs_og[age].copy().reset_index()
                
                stock_uecs[age].set_index(['region', 'AEO_industry',
                                            'ST_enduse'], inplace=True)
    
                stock_uecs[age] = \
                    stock_uecs[age][~stock_uecs[age].index.duplicated()]
                    
        # Use mask to multiply only where energy use by a fuel type occurs for
        # a given value of shipments.
        # returns dictionaries of dataframes that are indexed to region,
        # AEO_industry, and ST_enduse.
        # New_stock and old_stock are in terms of value of shipments.
        for col in new_stock.keys():

            for dfs in [(new_stock, initial_stock['new']),
                        (old_stock, initial_stock['old'])]:

                masked = np.ma.array(dfs[1][years], mask=dfs[0][col])

                dfs[0][col] = np.multiply(masked.mask, dfs[1][years])

        # This is the total stock that is used as a final weighting to
        # calculate the UECs. This is the original stock before any
        # adjustments from fuel switching.
        total_ft_stock = {}

        for ft in new_stock.keys():

            total_ft_stock[ft] = \
                pd.DataFrame(new_stock[ft][years].add(old_stock[ft][years]),
                            copy=True)

        # Calculate effect of UECS by fuel type and temperature range.
        # Note no current distiction.
        # Also, all ST_enduses that aren't proved a TPC by AEO/NEMS remain
        # unchanged (=1 throughout projection period)

        # This is where fuel switching takes place
        if fs_input == None:

            pass

        else:

            new_stock = fuel_switching.switch_fuels(
                initial_stock, new_stock, county_enduse, cls.fuel_types,
                fs_input['change_2050'], ind=fs_input['ind'],
                enduses=fs_input['eu'], 
                temp_band=fs_input['temp_band']
                )

        # This stock total is larger than the sum of new and old initial_stock
        # becuase value of shipments data are reapplied for each fuel type
        # (e.g., 2020 value of shipments for glass is applied to each instance
        # of use of a fuel type)

        uec_calc = pd.DataFrame()

        for ft in new_stock.keys():

            ft_stock_new = new_stock[ft][years[1:]].multiply(
                stock_uecs['new'][years[1:]], fill_value=0
                )

            ft_stock_old = old_stock[ft][years[1:]].multiply(
                stock_uecs['old'][years[1:]], fill_value=1
                )

            ft_stock_sum = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                list(itools.product([ft], years[1:])),
                names=('fuel_type', 'year')
                ), index=ft_stock_new.index)

            ft_stock_sum[ft] = ft_stock_new.add(ft_stock_old)

            uec_calc = \
                pd.concat([uec_calc, ft_stock_sum], axis=1)
                    
            uec_calc[ft] = uec_calc[ft].divide(total_ft_stock[ft], fill_value=1)
            
        uec_calc.fillna(1, inplace=True)

        if 'ST_enduse' not in county_enduse.index.names:
            
            county_enduse.set_index('ST_enduse', append=True, inplace=True)

#        uecs['weighted'].drop('ST_enduse', axis=1, inplace=True, level=0)

        return uec_calc

    @classmethod
    def Calc_Stock_Effects(cls, uecs_weighted, enduse,
                           energy_projections, VS, geo=None):

        """
        Must match enduse and energy_projections to the specified
        geo ('regional' or 'county'). Convert to MMBtu.
        """

        years = energy_projections.columns.levels[1][1:]
        
        if geo == 'county':

            levels_reset = [2, 4, 5]

        else:

            levels_reset = [2, 4]

        # set Total_energy_use as the sum of individual fuel types
        fts_sum = list(cls.fuel_types)

        fts_sum.remove('Total_energy_use')

        stock_turnover_effects = {}

        if geo == 'county':

            stock_turnover_effects['Total_energy_use'] = \
                pd.DataFrame(index=enduse.reset_index().set_index(
                    ['region', 'AEO_industry', 'ST_enduse', 'naics', 'fips_matching']
                    ).index, columns=uecs_weighted.columns.levels[1])
                    
            for col in uecs_weighted.columns.levels[0][:-1]:

                stock_turnover_effects[col] = energy_projections[col].reset_index(
                    level=levels_reset, drop=True
                    )[years].multiply(uecs_weighted[col].reindex(
                            index=enduse.reset_index(
                                level=levels_reset
                                ).index
                            )) * 1e6

                stock_turnover_effects[col].set_index(
                    [energy_projections.index.get_level_values('naics'),
                    energy_projections.index.get_level_values('fips_matching')],
                    append=True, inplace=True
                    )

        if geo == 'regional':

            stock_turnover_effects['Total_energy_use'] = \
                pd.DataFrame(index=enduse.reset_index().set_index(
                    ['region', 'AEO_industry', 'ST_enduse', 'naics']
                    ).index, columns=uecs_weighted.columns.levels[1])
                    
            for col in uecs_weighted.columns.levels[0][:-1]:
                
                stock_turnover_effects[col] = energy_projections[col].reset_index(
                    level=levels_reset, drop=True
                    )[years].multiply(uecs_weighted[col].reindex(
                            index=enduse.reset_index(level=2).index
                            )) * 1e6
                            
                stock_turnover_effects[col].set_index(
                    [enduse.index.get_level_values('naics')],
                    append=True, inplace=True
                    ) 

        for ft in fts_sum:

            stock_turnover_effects['Total_energy_use'] = \
                stock_turnover_effects['Total_energy_use'].add(
                    stock_turnover_effects[ft], axis=1, fill_value=0
                    )

        ste_pd =  pd.DataFrame()

        ste_cols = []

        for k in stock_turnover_effects.keys():
            
            ste_cols.append(list(itools.product([k], years)))

            ste_pd = pd.concat([ste_pd, stock_turnover_effects[k]], axis=1)

        ste_cols = [item for sublist in ste_cols for item in sublist]
        
        ste_pd.columns = pd.MultiIndex.from_tuples(ste_cols,
                                                   names=('fuel', 'year'))

        return ste_pd

    @classmethod
    def bw_efficiency(cls, bw_input, uec_weighted):
        """
        Calculates projected energy impacts of bandwidth efficiency. 
        Assumes a constant annual change in efficency from 2015 - 2050
        """

        bw_change = pd.DataFrame(index=bw_input.index,
                                 columns=range(2014, cls.proj_years[-1]+1))

        for i in bw_change.index:
            
            bw_change.loc[i, :] = \
                np.linspace(1, bw_input.ix[i].values,
                            len(range(cls.proj_years[0], cls.proj_years[-1])))

        bw_change = bw_change[cls.proj_years[1:]]

        uw_grpd = uec_weighted.reset_index().groupby('AEO_industry')

        uec_weighted.reset_index(inplace=True)

        for g in uw_grpd.groups:

            if (g not in bw_input.index.values) & \
                ('Balance of Manufacturing' not in bw_input.index.values): 

                continue

            if (g not in bw_input.index.values) & \
                ('Balance of Manufacturing' in bw_input.index.values):

                g = 'Balance of Manufacturing'

            else:

                pass
  
            uec_weighted.loc[
                uw_grpd.get_group(g).index, tuple(cls.proj_years[1:])
                ] =\
                    uw_grpd.get_group(g)[cls.proj_years[1:]].multiply(
                        bw_change.ix[g].values
                        )

        uec_weighted.set_index(['region', 'AEO_industry', 'ST_enduse'],
                               inplace=True, drop=True)

        return uec_weighted




    def industry_efficiency(AEO_industry, percent_increase_efficiency,
                            uec_weighted):
        """
        """
        industry = str(industry).lower()

        percent_increase_efficiency = float(percent_increase_efficiency)

        efficiency_df = pd.DataFrame(uec_weighted, copy=True)

        for i in efficiency_df.index:

            j = i[0]

            if str(j).lower() == AEO_industry:

                efficiency_2050 = \
                    (uec_weighted.loc[i, 2050] - percent_increase_efficiency/100)

                annual_efficiency = \
                    np.linspace(uec_weighted.loc[i, 2014], efficiency_2050,
                                                len(uec_weighted.columns))

                efficiency_df.loc[i] = annual_efficiency

            else:
                pass

        new_stock_turnover_energy_enduse = \
            regional_enduse.multiply(efficiency_df)

        return new_stock_turnover_energy_enduse, efficiency_df
