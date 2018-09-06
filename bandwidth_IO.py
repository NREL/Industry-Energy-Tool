# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:13:36 2018

@author: cmcmilla
"""
import pandas as pd
import numpy as np


class Bandwidth(object):
    """
    Class for importing, formatting, and matching energy efficiency bandwidth
    data.
    """

    def __init__(self):

        self.data_dir = \
            "Y:\\6A20\\Public\\ICET\\Data for calculations\\Default inputs\\"

        self.bw_file = "Bandwidth_data.xlsx"

        self.bw = pd.read_excel(self.data_dir + self.bw_file,
                                sheetname='import')

        self.bw.loc[:, 'nN'] = self.bw.NAICS.apply(lambda x: len(str(x)))

        self.bw_grpd = self.bw.groupby('nN')

        for g in self.bw_grpd.groups:

            for n in range(3, g + 1):
                self.bw.loc[self.bw_grpd.get_group(g).index, 'N' + str(n)] = \
                    [float(str(x)[0:n]) for x in self.bw.loc[
                        self.bw_grpd.get_group(g).index, 'NAICS'
                        ]]            
        # Create dictionaries based 3-digit NAICS (name and code) mapped
        # to NAICS associated with bandwidth studies. 
        def make_n_dict(df, col):

            n_dict = {}

            bw_n_grpd = df.groupby(col)

            for g in bw_n_grpd.groups:

                n_dict[g] = list(bw_n_grpd.get_group(g).NAICS)

            return n_dict

        self.bw_n_dict = {}

        self.bw_n_dict['name'] = make_n_dict(self.bw, '3dN_desc')

        self.bw_n_dict['code'] = make_n_dict(self.bw, 'N3')
        
    def bw_Nmatch(self, county_enduse, regional_enduse):
        """
        Match county-level NAICS to bandwidth results.
        """

        for df in [county_enduse, regional_enduse]:
            df.reset_index(inplace=True)

        enduse_naics = pd.DataFrame(regional_enduse.naics.drop_duplicates(),
                                    dtype='int')

        bw_n = pd.DataFrame(self.bw[['NAICS', 'N3', 'N4', 'N5']])

        bw_n.rename(columns={'NAICS': 'N6'}, inplace=True)

        nctest = [enduse_naics.naics.apply(
            lambda x: int(str(x)[0:len(str(x)) - i])
            ) for i in range(0, 4)]

        nctest = pd.concat(nctest, axis=1, ignore_index=True)
    
        nctest.reset_index(inplace=True, drop=True)

        nctest.columns = ['N6', 'N5', 'N4', 'N3']

        nctest.loc[:, 'bw_naics'] = np.nan

        for c in ['N6', 'N5', 'N4', 'N3']:

            n_dict = dict(pd.merge(nctest[nctest.bw_naics.isnull()], bw_n,
                left_on=nctest[nctest.bw_naics.isnull()][c], right_on=bw_n[c],
                how='inner')[[c + '_x', c + '_y']].values)

            nctest.loc[nctest[nctest.bw_naics.isnull()].index, 'in'] = \
                [x in n_dict.keys() for x in nctest[
                    nctest.bw_naics.isnull()
                    ][c]]

            cn_grpd = nctest[nctest.bw_naics.isnull()].groupby('in')

            nctest.loc[cn_grpd.get_group(True).index, 'bw_naics'] = \
               cn_grpd.get_group(True)[c].map(n_dict)

        # loop wasn't working here for some reason.
        #for df in [county_enduse, regional_enduse]:

        county_enduse = pd.merge(county_enduse, nctest[['N6', 'bw_naics']],
                                 left_on='naics', right_on='N6', how='left')

        county_enduse.drop('N6', axis=1, inplace=True)

        regional_enduse = pd.merge(regional_enduse, nctest[['N6', 'bw_naics']],
                                 left_on='naics', right_on='N6', how='left')

        regional_enduse.drop('N6', axis=1, inplace=True)

        regional_enduse.set_index(
            ['region', 'AEO_industry', 'naics', 'ST_enduse'], inplace=True,
            drop=True
            )

        county_enduse.set_index(
            ['region', 'AEO_industry', 'naics', 'ST_enduse', 'ALT_enduse',
            'fips_matching'], inplace=True, drop=True
            )
            
        # Create dictionary to map between bw_naics and original NAICS used for
        # bandwidth studies
            
        bw_master_dict = {}
        bw_m_df = pd.DataFrame()
        
#        z = {**x, **y}
        for col in ['N6', 'N5', 'N4', 'N3']:
            bw_m_df = bw_m_df.append(self.bw[[col, 'NAICS']].rename(columns={col:'bw_naics'}).dropna())
            
        bw_master_dict = dict(bw_m_df.values)

        return county_enduse, regional_enduse, bw_master_dict

    def set_bw_inputs(self, industry=['all'], scaling=100, reduction='PM_high'):
        """
        Define a set of year 2050 energy reduction for specified list of
        industries (3-digit NAICS, AEO industry, or 'all') 
        and reduction bandwidth (i.e., SOA_opp or PM_opp) and a
        scaling factor. 
        """
        
        bw_input = pd.DataFrame(index=self.bw.set_index('NAICS').index,
                                columns=['efficiency_2050'])

        if reduction == 'PM_high':

            bw_input.loc[:, 'efficiency_2050'] = \
                pd.concat([self.bw.set_index('NAICS')[reduction].dropna(),
                           self.bw.set_index('NAICS').PM_opp.dropna()], axis=0)

        else:

            bw_input.loc[:, 'efficiency_2050'] = \
                self.bw.set_index('NAICS')[reduction]

        if industry == ['all']:

            pass

        else:

            selected = pd.DataFrame()

            if (type(industry[0]) == np.str) & (industry != ['all']):

                for i in industry:
                    selected = pd.concat(
                        [selected, pd.Series(
                            [x in bw_n_dict['name'][i] for x in bw_input.index]
                            )], axis=1
                        )

                selected.set_index(bw_input.index, inplace=True)

                selected.replace({False: np.nan}, inplace=True)

            else:

                for i in industry:
                    selected = pd.concat(
                        [selected, pd.Series(
                            [x in bw_n_dict['code'][i] for x in bw_input.index]
                            )], axis=1
                        )

                selected.set_index(bw_input.index, inplace=True)

                selected.replace({False: np.nan}, inplace=True)

            bw_input = bw_input.ix[selected.dropna(how='all').index]

        bw_input.loc[:, 'efficiency_2050'] = \
            1 - bw_input.efficiency_2050.apply(lambda x: x * scaling / 100)

        return bw_input
