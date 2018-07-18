import pandas as pd

import numpy as np

class Ag(object):
    """
    Estimate 2014 county-level data based on 2012 USDA Census results.
    Only diesel, gasoline, LP gas, and 'other' (assumed to be residual
    fuel oil) are included.
    """

    @staticmethod
    def import_format(energy, ag_file):
        """
        Import calculated state and USDA-region electricity, diesel,
        gasoline, LPG, and 'other' fuel use by NAICS code for 2014. 
        Note that GHGRP includes 3 agricultural facilities.
        """

        if energy == 'electricity':
            e_tab = 'Electricity_MMBtu_2014'

        else:
            e_tab = 'Fuels_MMBtu_2014'

        ag_state_TBtu = pd.read_excel(ag_file, 
            sheetname = e_tab, header = 0, skiprows = [0,1]
            )

        ag_state_TBtu = pd.DataFrame(ag_state_TBtu.iloc[:, 0:52])

        ag_state_TBtu.dropna(axis = 0, how = 'all', inplace = True)

        ag_state_TBtu["NAICS"] = ag_state_TBtu.loc[:, "NAICS"].apply(
            lambda x: x.split('(')[1].split(')')[0]
            )

        if energy != 'electricity':
            ag_state_TBtu['Fuel'] = ag_state_TBtu.loc[:, "Fuel"].apply(
                lambda x: x.split('-')[0].split(',')[1].strip()
                )

        #Convert to TBtu and reformat
        ag_state_TBtu.iloc[:, 2:53] = ag_state_TBtu.iloc[:, 2:53] / 1E6

        ag_state_TBtu = pd.melt(ag_state_TBtu, 
            id_vars = ag_state_TBtu.columns.values[0:2], var_name = 'State', 
                value_name = 'Energy_TBtu'
            )

        return ag_state_TBtu

        #It's possible to use the USDA API to download data from Ag Census and 
        #Surveys.

        # class USDA_APIObject(object):
        #   """
        #   This product uses the NASS API but is not endorsed or certified by 
        #   NASS.
        #   Object for grabbing county-level 2012 Ag Census data.
        #   """

        #   def __init__(self, api_url, api_key):
        #       self.url = '%s?api_key=%s' % (api_url, api_key)

        #   @classmethod
        #   def get_datat(payload):
        #       self.response = requests.get(self.url)

        #       source_desc = 'CENSUS' sector_desc = 'ECONOMIC', group_desc = 'FARMS & LAND & ASSETS', short_desc = 'FARM OPERATIONS - NUMBER OF OPERATIONS'
        #           year = 2012, agg_level_desc = 'COUNTY', domain_desc = 'NAICS CLASSIFICATION'

    @staticmethod   
    def county_counts_calc(farm_counts_file):
        """"
        Read and format 2012 Ag Census data for county-level counts of 
        farms by NAICS.
        """
        
        ag_county_counts = pd.read_csv(
            farm_counts_file, low_memory = False
            )

        ag_county_counts['Domain Category'] = \
            ag_county_counts.loc[:, 'Domain Category'].apply(
                lambda x: x.split('(')[1].split(')')[0]
                )
        #Remove observations for NAICS 1119, which double counts 
        #observations for 11192 and "11193 & 11194 & 11199".
        ag_county_counts = pd.DataFrame(
            ag_county_counts[ag_county_counts['Domain Category'] != '1119']
            )

        #Calculate the fraction of county-level establishments by NAICS.    
        ag_state_counts = pd.pivot_table(
            ag_county_counts, values = ['Value'], index = ['State'], 
                columns = ['Domain Category'], aggfunc = np.sum
            )

        for s in ag_state_counts.index:

            for n in ag_state_counts.columns.levels[1]:

                i = ag_county_counts[(ag_county_counts.State == s) & \
                    (ag_county_counts['Domain Category'] == n)].index

                if i.size == 0:
                    pass

                else:
                    ag_county_counts.loc[i, 'StateFraction'] = \
                        ag_county_counts[
                            ag_county_counts.State == s
                        ].Value / ag_state_counts.loc[s, ('Value', n)]

        return ag_county_counts

    @staticmethod
    def county_energy_calc(ag_county_counts, ag_file):
        """
        Calculates county-level energy in TBtu based on county farm counts 
        by NAICS and state-level energy by NAICS.   
        """
        ag_county_energy = pd.DataFrame(
            ag_county_counts[[
                'State ANSI', 'County ANSI', 'Domain Category', 
                    'StateFraction'
                ]]
            )

        ag_county_energy.rename(columns = {'Domain Category': 'NAICS'},
            inplace = True)
            
        def import_format(energy, ag_file):
            """
            Import calculated state and USDA-region electricity, diesel,
            gasoline, LPG, and 'other' fuel use by NAICS code for 2014. 
            Note that GHGRP includes 3 agricultural facilities.
            """
    
            if energy == 'electricity':
                e_tab = 'Electricity_MMBtu_2014'
    
            else:
                e_tab = 'Fuels_MMBtu_2014'
    
            ag_state_TBtu = pd.read_excel(ag_file, 
                sheetname = e_tab, header = 0, skiprows = [0,1]
                )
    
            ag_state_TBtu = pd.DataFrame(ag_state_TBtu.iloc[:, 0:52])
    
            ag_state_TBtu.dropna(axis = 0, how = 'all', inplace = True)
    
            ag_state_TBtu["NAICS"] = ag_state_TBtu.loc[:, "NAICS"].apply(
                lambda x: x.split('(')[1].split(')')[0]
                )
    
            if energy != 'electricity':
                ag_state_TBtu['Fuel'] = ag_state_TBtu.loc[:, "Fuel"].apply(
                    lambda x: x.split('-')[0].split(',')[1].strip()
                    )
    
            #Convert to TBtu and reformat
            ag_state_TBtu.iloc[:, 2:53] = ag_state_TBtu.iloc[:, 2:53] / 1E6
    
            ag_state_TBtu = pd.melt(ag_state_TBtu, 
                id_vars = ag_state_TBtu.columns.values[0:2],
                    var_name='State', value_name='Energy_TBtu'
                )
    
            return ag_state_TBtu

        for sheet in ['electricity', 'fuels']:
            ag_state_TBtu = import_format(sheet, ag_file)

            for f in ag_state_TBtu.Fuel.drop_duplicates():
                state_fuel = pd.DataFrame(
                    ag_state_TBtu[ag_state_TBtu.Fuel == f]
                    )

                for n in ag_state_TBtu.NAICS.drop_duplicates():

                    energy_dict = dict(
                        state_fuel[state_fuel.NAICS == n][
                            ['State', 'Energy_TBtu']
                            ].values
                        )
                    
                    cty_index = ag_county_energy[
                        ag_county_energy['NAICS'] == n
                            ].index

                    ag_county_energy.loc[cty_index, f] = \
                        ag_county_energy.loc[cty_index, 'StateFraction'] * \
                            ag_county_energy.loc[cty_index, 'State ANSI'].map(
                                energy_dict
                                )       
    
        return ag_county_energy
