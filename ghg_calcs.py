# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:38:00 2017

@author: vnarwade
revised by cmcmilla
"""

import pandas as pd
import numpy as np
import itertools as itools
import re

class GHGs(object):
    """
    Class containing methods used to calculate electricity emission factors
    and GHG emissions for a single year.
    """

    data_wd = 'Y:\\6A20\\Public\\ICET\\Data for calculations\\GHGs\\'

    emission_factor_file = data_wd + 'Emission factors.xlsx'

    fips_to_zip_file = data_wd + 'COUNTY_ZIP_032014.xlsx'
    
    zip_to_census_region_file = data_wd + 'zip_census_region.csv'

    e_grid_file = data_wd + 'power_profiler_zipcode_tool_2014_v7.1_1.xlsx'

    aeo_electricity_supply_file = data_wd + 'AEO2017_sup_elec.xlsx'

    fips_census_div_file = data_wd + 'CountyFIPS_CensDiv.csv'

    county_population_file = data_wd + 'county_population.csv'

    county_energy_file = 'Y:\\6A20\\Public\\ICET\\Data for calculations\\' + \
        'Data foundation\\County_IndustryDataFoundation_2014_update_' + \
        '20170910-0116.csv'

    fips_zip_dict = \
        dict(pd.read_excel(fips_to_zip_file,
                           sheetname='COUNTY_ZIP_032014').values)

    fips_census_div_dict = dict(
        pd.read_csv(fips_census_div_file,
                    usecols=['FIPS', 'Census_Division']).values
        )

    zip_census_region_dict = \
        dict(pd.read_csv(zip_to_census_region_file).values)
    
    county_population_dict = dict(
        pd.read_csv(county_population_file,
                    usecols=['FIPS', 'Population Estimate 2014']).values
        )

    zip_sub_region = pd.read_excel(e_grid_file, sheetname='Zip-subregion')

    zip_sub_region.fillna(0, inplace=True)

    fuel_efs = pd.read_excel(
        emission_factor_file, sheetname='emission_factors', index_col=[0]
        )

    @classmethod
    def elect_emission_factors(cls, perc_RE_incr=None):
        """
        Method for calculating the total electricity emission factor (including
        CH4 and N2O) by zip code. Based on data from EPA's eGRID. Also
        includes option for increasing RE %-pt by 2050 (dicionary of region: increase.
        Currently only one region or all regions).
        """
        # Calculate projected emissions and % RE values from AEO

        nems_egrid_regions = pd.read_csv(cls.data_wd + 'nems_egrid_regions.csv')

        aeo_emm = pd.read_excel(cls.aeo_electricity_supply_file, 
                                sheetname='sup_elec.1208a',
                                index_col=False)

        agg_regions = ['Northeast Power Coordinating Council /', 
                'Western Electricity Coordinating Council /']

        def find_nems_region(text):

            search_string = "\d+ - (\w+ )+/ (\w+ )+(\w+)|\d+ - (\w+ )+/ (\w+)|\d+ -( \w+)"
            try:
                re.search(search_string, text)

            except TypeError:
                aeo_reg = np.nan

            else:
                if re.search(search_string, text) == None:
                    aeo_reg = np.nan

                else:
                    aeo_reg = \
                        re.split(' - ',
                            re.search("\d+ - (\w+ )+/ (\w+ )+(\w+)|\d+ - (\w+ )+/ (\w+)|\d+ - (\w+ )+(\w+)",
                                text).group()
                        )[1]

            return aeo_reg

        aeo_emm.loc[:, 'reg_match'] = aeo_emm[2015].apply(
            lambda x: find_nems_region(x)
            )

        aeo_emm.reg_match.fillna(method='ffill', inplace=True)

        aeo_emm = aeo_emm[aeo_emm.reg_match.notnull()]

        aeo_emm.dropna(axis=0, thresh=5, inplace=True)

        aeo_emm.drop([aeo_emm.columns[0]], axis=1, inplace=True)

        aeo_emm.loc[:, 2015] = aeo_emm[2015].apply(lambda x: x.lstrip())

        aeo_emm.replace({'- -': np.nan}, inplace=True)

        new_cols = ['desc']

        [new_cols.append(c) for c in aeo_emm.columns]

        new_cols.remove('Unnamed: 37')

        aeo_emm.columns = new_cols

        # Correct SERC region
        aeo_emm.reg_match.replace(
            {'SERC Reliability Corporation / Virginia': 'SERC Reliability Corporation / Virginia-Carolina',
             'Northeast Power Coordinating Council / NYC': 'Northeast Power Coordinating Council / NYC-Westchester'
            }, inplace=True
            )

        aeo_emm_grpd = aeo_emm.groupby(['reg_match', 'desc'])

        aeo_ef_re = pd.DataFrame()

        def calc_aeo_data(reg, data_field):

            if data_field == 'RE':

                data_column = 'Renewable Sources 14/'

                data_name = 'RE_perc'

            if data_field == 'EF':

                data_column = 'Carbon Dioxide (million short tons)'

                data_name = 'CO2eq_kg_perTBtu'

            df = pd.DataFrame({'value': aeo_emm_grpd.get_group((reg, data_column)).sum()[1:-1].divide(
                aeo_emm_grpd.get_group((reg, 'Total Electricity Generation')).sum()[1:-1], 
                fill_value=0
                )})

            # Convert from Million tons/TWh to kg CO2/TBtu
            if data_field =='EF':

                df.loc[:, 'value'] = df.value.divide(3.41214163) * 1e9

            df.reset_index(inplace=True)

            df.rename(columns={'index': 'year'}, inplace=True)

            df.loc[:, 'data'] = data_name

            df.loc[:, 'EMM_region'] = reg

            return df

        for reg in aeo_emm.reg_match.drop_duplicates():

            df = calc_aeo_data(reg, 'RE')

            aeo_ef_re = aeo_ef_re.append(df, ignore_index=True)

            df = calc_aeo_data(reg, 'EF')

            aeo_ef_re = aeo_ef_re.append(df, ignore_index=True)

        aeo_ef_re[aeo_ef_re.data != 'RE_perc']

        aeo_ef_re = pd.merge(aeo_ef_re, nems_egrid_regions,
                             left_on='EMM_region',
                             right_on='AEO', how='outer')

        def increase_RE(aeo_ef_re, increase, region='all'):

            years = aeo_ef_re.year.drop_duplicates()

            steps = years[-1:] - years[0] + 2

            increase = increase / 100

            ann_increase = pd.Series(
                np.linspace(0, increase, num=steps,endpoint=True)[1:],
                name='value'
                )

            ann_increase = pd.concat([ann_increase, years], axis=1)

            data = aeo_ef_re.copy(deep=True)

            if region == 'all':

                pass

            else:

                data = data[data.eGRID == region]

                print(data.head())

            def adj_values(data, ann_increase, re_ef):

                if re_ef == 'RE':

                    new_values = data[(data.data == 'RE_perc')].copy()  

                if re_ef == 'ef':

                    new_values = data[(data.data == 'CO2eq_kg_perTBtu')].copy()   

                new_values.reset_index(inplace=True)

                new_values.sort_values(['EMM_region', 'year'], inplace=True)

                new_values.set_index('EMM_region', inplace=True)

                for i in new_values.index.drop_duplicates():

                    if re_ef == 'RE':

                        new_values.loc[i, 'value'] = \
                            new_values.loc[i, 'value'].add(
                                ann_increase.value.values
                                ).values

                    if re_ef == 'ef':

                        new_values.loc[i, 'value'] = \
                            new_values.loc[i, 'value'].multiply(
                                1 - ann_increase.value.values
                                ).values

                new_values.reset_index(inplace=True)

                new_values.set_index('index', inplace=True)

                if re_ef == 'RE':

                    new_values.loc[new_values[new_values.value > 1].index, 'value'] = 1

                return new_values

            for re_ef in ['RE', 'ef']:

                df = adj_values(data, ann_increase, re_ef)

                data.loc[df.index, 'value'] = df.value

            return data

        if perc_RE_incr is not None:

            for k, v in perc_RE_incr.items():

                aeo_ef_re.update(
                    increase_RE(aeo_ef_re, v, region=k)
                    )

        else:

            pass

        # Calcualte eGRID emission factors by zip code.
        zip_single_subregions = pd.DataFrame(
            cls.zip_sub_region[
                cls.zip_sub_region['eGRID Subregion #2'] == 0
                ], copy=True
            )

        zip_multiple_subregions = pd.DataFrame(
            cls.zip_sub_region[
                cls.zip_sub_region['eGRID Subregion #2'] != 0
                ], copy=True
            )

        # This should be refactored into a method for single subregions (sr)
        # and multiple subregions (mr)
        ef_sr = pd.read_excel(cls.e_grid_file,
                           sheetname='eGRID Subregion Emission Factor',
                           skiprows=[0, 1 ,2])

        match_col_sr = 'Subregion'

        ef_mr = pd.read_excel(cls.e_grid_file, sheetname='Data Entry',
                            skiprows=[0, 1], 
                            parse_cols="B,T:V" )

        ef_mr.columns = ['Zip', 'SRCO2RTA', 'SRCH4RTA', 'SRN2ORTA']

        match_col_mr = 'Zip'

        for df in [ef_sr, ef_mr]:
 
            df.loc[:, 'CO2eq_kg_perTBtu'] = \
                    pd.Series((df.SRCO2RTA + df.SRCH4RTA * 25 +
                                df.SRN2ORTA * 298) * 
                                (0.453592 / 3.412 * 10**6))

        ef_mr_dict = dict(zip(ef_mr.iloc[:, 0], ef_mr['CO2eq_kg_perTBtu']))

        z_msr_grpd = zip_multiple_subregions.groupby(
            ['eGRID Subregion #1', 'eGRID Subregion #2',
             'eGRID Subregion #3'])

        z_mean = pd.DataFrame()

        for g in z_msr_grpd.groups:

            mr = pd.DataFrame()

            if True in [h in ['AKMS', 'AKGD', 'HIOA', 'HIMS'] for h in g]:

                continue

            for r in g:

                if r == 0:

                    continue

                else:

                    mr = mr.append(
                        aeo_ef_re[aeo_ef_re.data == 'CO2eq_kg_perTBtu'].pivot(
                            index='eGRID', columns='year', values='value'
                            ).ix[r]
                        )

            mr_mean = mr.mean().values.reshape(mr.shape[1], 1)

            mr_mean = \
                pd.DataFrame(np.tile(mr_mean.T,
                                    (len(z_msr_grpd.get_group(g).index), 1)),
                             index=[z_msr_grpd.get_group(g).index],
                             columns=mr.mean().index)

            z_mean = z_mean.append(mr_mean)

        zip_multiple_subregions = pd.merge(zip_multiple_subregions, z_mean,
                                           left_index=True, right_index=True,
                                           how='left')

        zip_multiple_subregions.loc[:, 2014] =\
            zip_multiple_subregions.iloc[:, 0].map(ef_mr_dict)


        def ak_hi_RE(df, increase, region):
            """
            Calculates change in electricity emission factors (efs) for AK and 
            HI. Assumes decrease in efs is proportional to increase in RE.
            Note no projected efs from AEO; projections instead based on 
            eGRID values for 2014.
            """

            years = [y for y in range(df.columns[6], df.columns[-2] + 1)]

            steps = years[-1] - years[0] + 2

            increase = increase / 100

            ann_decrease = 1 - pd.Series(
                np.linspace(0, increase, num=steps, endpoint=True)[1:],
                name='value'
                )

            ann_decrease = \
                ann_decrease.values.reshape(ann_decrease.shape[0], 1).T

            def test_region(df, region):

                if region == 'all':

                    ak_hi_index = df.query('state == ["AK", "HI"]').index

                elif True in [r in ['AKMS', 'AKGD'] for r in region]:

                    ak_hi_index = df.query('state == ["AK"]').index

                elif True in [r in ['HIOA', 'HIMS'] for r in region]:

                    ak_hi_index = df.query('state == ["HI"]').index

                elif (True in [r in ['HIOA', 'HIMS'] for r in region]) and \
                    (True in [r in ['AKMS', 'AKGD'] for r in region]):

                    ak_hi_index = df.query('state == ["AK", "HI"]').index

                else:

                    ak_hi_index = pd.Series(False)

                return ak_hi_index

            ak_hi_index = test_region(df, region)

            if ak_hi_index.all() == True:

                ann_decrease = \
                    pd.DataFrame(np.tile(ann_decrease, (len(ak_hi_index), 1)),
                                 index=[ak_hi_index], columns=years)

                ann_decrease = \
                    ann_decrease.multiply(df.loc[ak_hi_index, 2014], axis=0)

                df.update(ann_decrease)

                return df               

            else:

                pass

        ef_sr_dict = dict(zip(ef_sr.iloc[:, 1], ef_sr['CO2eq_kg_perTBtu']))

        zip_single_subregions = pd.merge(
            zip_single_subregions,
            aeo_ef_re[aeo_ef_re.data == 'CO2eq_kg_perTBtu'].pivot(
                index='eGRID', columns='year', values='value'
                ), left_on=['eGRID Subregion #1'], right_index=True,
            how='outer'
            )

        zip_single_subregions.loc[:, 2014] =\
            zip_single_subregions.iloc[:, 3].map(ef_sr_dict)

        # Calculate changes to AK and/or HI emission factors, if applicable.
        if perc_RE_incr is not None:

            for df in [zip_multiple_subregions, zip_single_subregions]:

                for k, v in perc_RE_incr.items():

                    df = ak_hi_RE(df, v, k)

        for df in [zip_multiple_subregions, zip_single_subregions]:

            df.drop(df.columns[1:6], axis=1, inplace=True)

            df.rename(columns={'ZIP (character)': match_col_mr}, inplace=True)

        electric_efs = zip_single_subregions.append(zip_multiple_subregions)

        electric_efs = electric_efs.fillna(method='bfill', axis=1)

        electric_efs.set_index('Zip', inplace=True)

        electric_efs.sort_index(axis=1, inplace=True)

        electric_efs.reset_index(inplace=True)

        electric_efs.loc[:, 2013] = electric_efs[2014]

        return electric_efs

    @classmethod
    def Calc_Baseline_GHGs(cls, county_enduse, electric_efs, enduse=False):
        """
        Method for calculating GHGs for a single year by county based on
        whether energy by end use is specified.
        Output is a dictionary summarizing emissions on various geographic
        scales.
        """
        
        if type(county_enduse.index) == pd.indexes.multi.MultiIndex:
            county_enduse.reset_index(inplace=True)

        fuel_types = ['Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL',
                      'Residual_fuel_oil', 'Net_electricity', 'Natural_gas',
                      'Other']

        other_cols = ['<100', '100-249', '250-399', '400-999', '>1000',
                    'ST_enduse']

        geo_columns = ['fips_matching', 'naics', 'fipscty', 'fipstate',
                       'subsector', 'region']

        if enduse is True:

            e_columns =\
                [item for sublist in
                    [other_cols, geo_columns]
                    for item in sublist]

        else:

            e_columns = [item for sublist in [geo_columns] for item in sublist]

        emissions = pd.DataFrame(county_enduse, columns=e_columns)

        emissions.loc[:, 'Zip'] = emissions.fips_matching.map(
            cls.fips_zip_dict
            )

        emissions.loc[:, 'Census_div'] = \
            county_enduse['fips_matching'].map(cls.fips_census_div_dict)

        emissions.loc[:, 'electricity_ef'] = \
            emissions['Zip'].map(
                dict(electric_efs[['Zip', 2014]].values)
                )

        emissions.loc[:, 'electric_emissions'] = \
            county_enduse.Net_electricity.multiply(
                emissions.electricity_ef
                )

        for f in cls.fuel_efs.index:

            if f == 'Petrol':

                pass

            else:

                col_name = f + '_emissions'

                emissions.loc[:, col_name] = \
                    county_enduse[f].multiply(cls.fuel_efs.ix[f][0])

        emission_columns = ['electric_emissions', 'Diesel_emissions',
                            'Coal_emissions', 'Coke_and_breeze_emissions',
                            'LPG_NGL_emissions', 'Natural_gas_emissions',
                            'Residual_fuel_oil_emissions',
                            'Other_emissions']

        emissions['Total_emissions'] = \
            emissions[emission_columns].sum(axis=1)

        # summarize emissions on various geographic scales.
        emissions_summary = {}

        emissions_summary['county_MTCO2e'] = emissions.groupby(
            'fips_matching'
            ).sum().divide(1000)

        emissions_summary['national_MMTCO2e'] = \
            emissions[emission_columns].sum(axis=0).divide(1e9)

        emissions_summary['subsector_MMTCO2e'] = emissions.groupby(
            'subsector'
            )[emission_columns].sum().divide(1e9)

        emissions_summary['census_div_MMTCO2e'] = emissions.groupby(
            'Census_div'
            )[emission_columns].sum().divide(1e9)

        temp_emissions = pd.DataFrame(columns=other_cols[:-1], index=[0])

        for col in temp_emissions.columns:
            temp_emissions.loc[:, col] = \
                emissions[emissions[col].notnull()][
                    'Total_emissions'
                    ].sum() / 1e9

        emissions_summary['temp_MMTCO2e'] = temp_emissions

        emissions_summary['enduse_MMTCO2e'] = \
            emissions.groupby('ST_enduse')[emission_columns].sum().divide(1e9)

        return emissions_summary

    @classmethod
    def Calc_Proj_GHGs(cls, og_elect_ef, energy_proj, geo='county',
                        csv_exp=False, fname=None):
        """
        Calculate the GHG emissions of energy projections on regional or county
        level. Option to export results to working drive. Returns dataframe
        only if export option == False.
        Returns values in million metric tons CO2-eq (MMTCO2e)
        """

        fuel_types = ['Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL',
                      'Residual_fuel_oil', 'Net_electricity', 'Natural_gas',
                      'Other']

        if type(energy_proj) == dict:
            
            proj_emissions = \
                pd.DataFrame(index=energy_proj['Total_energy_use'].index)
                
        else:

            proj_emissions = pd.DataFrame(index=energy_proj.index)

        if geo == 'regional':

            elect_ef = pd.concat(
                [og_elect_ef, og_elect_ef.Zip.map(cls.zip_census_region_dict)],
                axis=1
                )

            elect_ef.columns.values[-1] = 'region' 

            elect_ef = \
                elect_ef.dropna().drop('Zip', axis=1).groupby('region').mean()

        if geo == 'county':

            f_z_df = pd.read_excel(cls.fips_to_zip_file,
                                   sheetname='COUNTY_ZIP_032014')

            f_z_df.rename(columns={'ZIP': 'Zip', 'COUNTY': 'fips_matching'},
                          inplace=True)

            f_z_df = pd.merge(f_z_df, og_elect_ef, on='Zip', how='left')

            f_z_df = f_z_df.dropna().drop('Zip', axis=1).groupby(
                'fips_matching'
                ).mean()

            if type(energy_proj) == dict:
                
                elect_ef = pd.merge(pd.DataFrame(
                    energy_proj['Total_energy_use'].index.get_level_values(
                        'fips_matching'
                        ).drop_duplicates()
                    ), f_z_df.reset_index(), on='fips_matching', how='left')

                
            else:
                
                elect_ef = pd.merge(pd.DataFrame(
                    energy_proj.index.get_level_values(
                        'fips_matching'
                        ).drop_duplicates()
                    ), f_z_df.reset_index(), on='fips_matching', how='left')

        ft_cols = []

        for ft in fuel_types:

            if ft != 'Net_electricity':

                proj_emissions = pd.concat(
                    [proj_emissions,
                     energy_proj[ft].multiply(
                         cls.fuel_efs.ix[ft][0], fill_value=0
                         )], axis='columns'
                    )

            if ft == 'Net_electricity':

                if geo == 'regional':

                    proj_emissions = pd.concat(
                        [proj_emissions,
                         energy_proj[ft].multiply(elect_ef, axis='index',
                                                  level=0).dropna(axis=1,
                                                                  how='all')],
                        axis='columns'
                        )

                if geo == 'county':

                    proj_emissions = pd.concat(
                        [proj_emissions,
                         energy_proj['Net_electricity'].multiply(
                            elect_ef.set_index('fips_matching'), axis='index',
                            level='fips_matching').dropna(axis=1, how='all')],
                         axis=1)

            # need to unpack this to rename columns correctly   
            ft_cols.append(
                list((itools.product([ft], energy_proj[ft].columns)))
                )

            # proj_emissions.rename(columns={ft: ft + '_emissions'},
            #                       inplace=True)

        ft_cols = [item for sublist in ft_cols for item in sublist]

        proj_emissions.columns = pd.MultiIndex.from_tuples(ft_cols,
                                                   names=('fuel', 'year'))

        proj_emissions = proj_emissions.divide(1e15)

        if 'ALT_enduse' in proj_emissions.index.names:
            proj_emissions.reset_index(level=['ALT_enduse'], drop=True,
                                       inplace=True)

        if csv_exp is True:

            proj_emissions.to_csv(geo + '_' + fname + '_ghg_proj',
                                  chunksize=5000, compression='gzip')

        else:

            return proj_emissions
