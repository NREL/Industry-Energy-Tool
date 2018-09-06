# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:16:18 2018

@author: cmcmilla
"""

import pandas as pd
import numpy as np
import os

#from Calc_EndUse_AllInd import EnergyEndUse


class TempChar(object):
    """
    Class for characterizing process temperatures (process heat and boilers).
    """

    file_dir = "Y:\\6A20\\Public\\ICET\\Data for calculations\\Temperature\\"

    nfiles = {'sic_N02_file': '1987_SIC_to_2002_NAICS.xls',
              'N02_N07_file': '2002_to_2007_NAICS.xls',  
              'N07_N12_file': '2007_to_2012_NAICS.xls'}

    temp_file = 'ProcessTemps_108.xlsx'

    eu_map = {'Process Heating': 'fuel',
              'Conventional Boiler Use': 'boiler',
              'CHP and/or Cogeneration': 'boiler'}

    @classmethod
    def ImportTemps(cls, mfg_enduse):
        """
        Import temperature data obtained from 108 Industrial Processes.
        Requires adjusting from 1997 SIC to 2002 NAICS to 2007 NAICS to 
        2012 NAICS.
        """

        def create_dict(file_dir, file):
            dict_out = dict(pd.read_excel(
                file_dir + file, _sheetname=0).iloc[:, (0, 2)].values
                )

            return dict_out

        ndict = {}

        for k, v in cls.nfiles.items():
            ndict[k[0:7]] = create_dict(cls.file_dir, v)

        temps = pd.read_excel(cls.file_dir + cls.temp_file, sheetname=0)

        temps.SIC.fillna(method='ffill', inplace=True)

        temps.loc[:, 'Temp_C'] = temps.Temp_C.apply(
            lambda x: int(np.around(x))
            )

        # Calculate energy fraction of each process by temperature
        temps = pd.DataFrame(temps.groupby(
            ['SIC', 'Unit_Process', 'Heat_type', 'Temp_C']
            )['E_Btu'].sum())

        e_totals = temps.reset_index()[
            temps.reset_index()['Unit_Process'] != 'Boiler'
            ].groupby(['SIC', 'Heat_type']).E_Btu.sum()

        for i in temps.index:
            if 'Boiler' in i:
                continue

            temps.loc[i, 'Fraction'] = \
                temps.loc[i, 'E_Btu'] / e_totals.loc[(i[0], i[2])]

        temps.reset_index(inplace=True)

        temps.loc[:, 'SIC'] = temps.SIC.apply(lambda x: int(str(x)[0:4]))

        temps.loc[:, 'NAICS02'] = temps.SIC.map(ndict['sic_N02'])

        temps.loc[:, 'NAICS07'] = temps.NAICS02.map(ndict['N02_N07'])

        temps.loc[:, 'NAICS12'] = temps.NAICS07.map(ndict['N07_N12'])

        # Multiple entries for each SIC/NAICS; take simple mean.        
        temps = temps.groupby(
            ['NAICS12', 'Unit_Process', 'Heat_type']
            )[['E_Btu', 'Temp_C', 'Fraction']].mean()

        # Create 4-, and 3-digit NAICS table for matching 
        temps_NAICS = pd.DataFrame(index=temps.index.levels[0],
                                   columns=['N5', 'N4', 'N3']
                                   )

        for n in [5, 4, 3]:
            temps_NAICS.loc[:, 'N' + str(n)] = \
                [float(str(x)[0:n]) for x in temps_NAICS.index.values]

        temps_NAICS.reset_index(inplace=True)

        eu_naics = pd.DataFrame(
            mfg_enduse.naics.drop_duplicates().sort_values(ascending=True),
            copy=True
            )

        eu_naics.reset_index(inplace=True, drop=True)

        eu_naics.rename(columns={'naics':'NAICS12'}, inplace=True)

        for n in [5, 4, 3]:
            eu_naics.loc[:, 'N' + str(n)] = \
                [float(str(x)[0:n]) for x in eu_naics.NAICS12.values]

        # Match naics between end use data set and temperature info. 
        nmatch = pd.DataFrame()
        for column in temps_NAICS.columns:
            nmatch = pd.concat([nmatch, pd.Series(
                [x in temps_NAICS[column].values for x in eu_naics[
                    column
                    ].values]
                )], axis=1)

        nmatch.columns = eu_naics.columns

        nmask = pd.DataFrame()

        for c in nmatch.columns:

            nmask = pd.concat(
                [nmask, eu_naics[c].multiply(nmatch[c])],
                axis=1
                )

        nmask.replace({0:np.nan}, inplace=True)

        # Values of 0 indicate no matching temperature data, even at 3-digit 
        # level.
        nmask.N3.fillna(0, inplace=True)

        nmask.loc[:, 'TN_Match'] = nmask.apply(
            lambda x: int(list(x.dropna())[0]), axis=1
            )

        nmask.rename(columns={'NAICS12':'N6'}, inplace=True)

        nmask.loc[:, 'NAICS12'] = eu_naics.NAICS12

        # Merge matched NAICS values with end use energy data
        mfg_enduse = pd.merge(mfg_enduse,
                                  nmask[['NAICS12', 'TN_Match']], how='left',
                                  left_on='naics', right_on='NAICS12')

        mfg_enduse.drop('NAICS12', inplace=True, axis=1)

        # Merge temps and temps_NAICS for future operations by other NAICS
        temps.reset_index(inplace=True)

        temps = pd.merge(temps, temps_NAICS, left_on='NAICS12',
                         right_on='NAICS12', how='left')

        for tb, tr in {'<100': (0, 99), '100-249': (100, 249),
                   '250-399': (250, 399), '400-999': (400, 999),
                   '>1000': (1000, 3000)}.items():

            ti = temps[temps.Temp_C.between(tr[0], tr[1])].index

            temps.loc[ti, 'Temp_Bucket'] = tb

        return mfg_enduse, temps

    def heat_mapping(cls, temps, mfg_enduse, char=None):
        """
        Map heat use characteristics (e.g., temperature) to end use
        disggregation.
        """

        # Calculate temperatures and associated energy use only for boiler
        # and process heating end uses.
        # Does not address industries that were not matched to a 3-digit NAICS
        eu_grpd = mfg_enduse[mfg_enduse.Enduse.apply(
            lambda x: x in cls.eu_map.keys()
            )].groupby(['TN_Match', 'Enduse'])

        ndict = {6: 'NAICS12', 5: 'N5', 4: 'N4', 3: 'N3'}

        eu_energy_t = pd.DataFrame(columns=temps.Temp_Bucket.drop_duplicates())

        for g in eu_grpd.groups:

            # Base unquantifeid temp ranges on known averages, discarding
            # higher temperature buckets (i.e., >250 C)
            if g[0] == 0:

                t_e = temps[temps.Unit_Process != 'Boiler'].replace(
                    {'Heat_type': {'steam': 'boiler', 'water': 'boiler'}}
                    ).groupby(
                        ['Heat_type', 'Temp_Bucket']
                        )[['E_Btu']].sum()

                t_e.drop(['>1000', '400-999', '250-399'], inplace=True,
                         level='Temp_Bucket')

                e_totals = t_e.reset_index().groupby(
                    ['Heat_type']
                    ).E_Btu.sum()

                for i in t_e.index:

                    t_e.loc[i, 'Fraction'] = \
                        t_e.loc[i, 'E_Btu'] / e_totals.loc[i[0]]

                eu_energy_t = eu_energy_t.append(
                    eu_grpd.get_group(g).Total_energy_use.apply(
                        lambda x: x * t_e.xs((cls.eu_map[g[1]]),
                                             level=(0)).Fraction
                        )
                    )

            else:

                n_n = ndict[len(str(g[0]))]

                # Define 'steam' and 'water' heat types as equivalently
                # produced from a boiler.
                t_e = temps[temps.Unit_Process != 'Boiler'].replace(
                    {'Heat_type': {'steam': 'boiler', 'water': 'boiler'}}
                    ).groupby(
                        [n_n, 'Heat_type', 'Temp_Bucket']
                        )[['E_Btu']].sum()

                e_totals = t_e.reset_index().groupby(
                    [n_n, 'Heat_type']
                    ).E_Btu.sum()

                for i in t_e.index:

                    t_e.loc[i, 'Fraction'] = \
                        np.array(t_e.loc[i, 'E_Btu'] / e_totals.xs((i[0], i[1]),
                                                            level=(0, 1)))[0]

                eu_energy_t = eu_energy_t.append(
                    eu_grpd.get_group(g).Total_energy_use.apply(
                        lambda x: x * t_e.xs((g[0], cls.eu_map[g[1]]),
                                             level=(0, 1)).Fraction
                        )
                    )

        # Calculations were picking up 2 null columns for some reason.
        eu_energy_t = pd.DataFrame(eu_energy_t[eu_energy_t.columns.dropna()])

        mfg_enduse_t = pd.concat([mfg_enduse, eu_energy_t], axis=1)

        return mfg_enduse_t
