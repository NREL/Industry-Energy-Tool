import pandas as pd

import numpy as np

#class cons_files(object):
#
#    """
#    Defines the excel files and tabs (sheets) associated with
#    calculations of county-level construction energy use.
#    """
#    def __init__(self, cf):
#        self.cf = cf
#
#    d_cons_data_names = {
#        'state3D_emply': 'IPF results and calcs',
#        'cons_fuel_frac': 'cons_3DN_fuel_frac',
#        'cons_prices': 'Dollar_per_MMBtu_python', 'cons_gdp': 'bea_GDP_CAGR'
#        }
#
#    @classmethod
#    def load_files(cls, tab_name):
#        """
#        Creates a dataframe from specified excel file and tab (sheet)
#        """
#        cons_df = pd.read_excel(cls.cf, sheetname=tab_name)
#
#        return cons_df

class Cons(object):
    """
    Calculates county-level energy use for the construction sector.
    """
    #Create dataframe dictionary of excel data required for county-level
    #calculations.
    
    def cons_dict(cfile):

        d_cons_data_names = {
            'state3D_emply': 'IPF results and calcs',
            'cons_fuel_frac': 'cons_3DN_fuel_frac',
            'cons_prices': 'Dollar_per_MMBtu_python', 'cons_gdp': 'bea_GDP_CAGR'
            }

        def load_files(cfile, tab_name):
            """
            Creates a dataframe from specified excel file and tab (sheet)
            """
            cons_df = pd.read_excel(cfile, sheetname=tab_name)
    
            return cons_df

        excel_dict = dict()

        for k, v in d_cons_data_names.items():
            excel_dict[k] = load_files(cfile, v)

        # first calculate fuel and electricity expenditures by state,
        # 3-digit NAICS, and employment size using cons_ipf results.
        # Expenditures are in $1,000
        excel_dict['state3D_emply'].rename(columns={
                210: 'n1_4', 220: 'n5_9', 230: 'n10_19', 241: 'n20_49',
                242: 'n50_99', 251: 'n100_249', 252: 'n250_499',
                254: 'n500_999', 260: 'n1000'
                }, inplace=True)
    
        excel_dict['cons_fuel_frac'].index = \
            excel_dict['state3D_emply'].index
        
        return excel_dict

    @staticmethod     
    def state_energy_calc(excel_dict):
        """
        Calculate state-level energy use for the construction sector.
        """
        state_fuel_exp = pd.DataFrame()

        for f in ['Diesel', 'Natural_gas', 'LPG', 'Electricity']:
            i = [x + "_" + f for x in excel_dict['state3D_emply'].index]
            f_df = pd.DataFrame(
                excel_dict['state3D_emply'].multiply(
                    excel_dict['cons_fuel_frac'][f], axis="index") 
                )

            f_df.index = i

            f_df['State'] = [
                x.split('_')[0] for x in excel_dict['state3D_emply'].index
                ]

            state_fuel_exp = state_fuel_exp.append(f_df)

        state_fuel_exp.loc[:, 'State_FIPS'] = \
            state_fuel_exp['State'].map(dict(
                excel_dict['cons_fuel_frac'][['Geographic area name',
                    'fips']].values))

        # Estimate 2014 fuel quantities (in MMBtu) based on 2012 fuel prices 
        # by state, PADD, etc. and GDP growth from 2012 - 2014.
        state_fuel_exp['NAICS_3D'] = [
            x.split('_')[1] for x in state_fuel_exp.index
            ]

        state_fuel_exp['Fuel_type'] = [
            x.split('_')[2] for x in state_fuel_exp.index
            ]

        state_fuel_exp.loc[state_fuel_exp.Fuel_type == 'Natural', \
            'Fuel_type'] = 'Natural_gas'

        state_fuel_exp.loc[state_fuel_exp.Fuel_type == 'Other', \
            'Fuel_type'] = 'LPG'

        state_fuel_MMBtu = pd.DataFrame(state_fuel_exp, copy = True)

        excel_dict['cons_prices'].index = excel_dict['cons_prices'][
            'Geographic area name']

        #Calculate MMBtu value for purchases (in $1,000)
        #fuel_df = pd.DataFrame()
        for f in ['Diesel', 'Natural_gas', 'LPG', 'Electricity']:
            state_grouping = state_fuel_exp[
                state_fuel_exp.Fuel_type == f].loc[:,
                    ('n1_4'):('State')
                        ].groupby('State')      

            for name in state_grouping.groups.keys():
                price = excel_dict['cons_prices'].loc[name, f]      
                MMBtu = \
                    state_grouping.get_group(name).loc[:, ('n1_4'):('n1000')] \
                        / price * pow((excel_dict['cons_gdp'].loc[
                            name, 'Real_GDP_CAGR_2012_2014'
                            ] / 100 + 1), 2
                        ) * 1000
                #test_df = test_df.append(MMBtu)

                state_fuel_MMBtu.update(MMBtu, overwrite=True)

        return state_fuel_MMBtu

    #Create state-level establishment counts by 3-digit NAICS from 
    #2014 CBP data.
    @staticmethod
    def county_frac_calc(cbp_corrected):
        cons_cbp = cbp_corrected[(cbp_corrected.naics > 230000) & \
            (cbp_corrected.naics < 240000)][
                ['fipstate', 'fipscty', 'naics', 'est', 'n1_4', 'n5_9',
                    'n10_19','n20_49', 'n50_99', 'n100_249', 'n250_499',
                        'n500_999','n1000']
                ]

        cons_cbp['NAICS_3D'] = cons_cbp.naics.apply(
            lambda x: int(str(x)[0:3])
            )

        c_state_counts = cons_cbp.groupby(['fipstate', 'NAICS_3D'])[
            ['n1_4', 'n5_9', 'n10_19','n20_49', 'n50_99', 'n100_249', 
                'n250_499', 'n500_999','n1000']
            ].aggregate(np.sum)

        for c in cons_cbp.columns[4:13]:
            for f in cons_cbp.fipstate.drop_duplicates():
                for n in cons_cbp.NAICS_3D.drop_duplicates():
                    frac_index = cons_cbp[
                        (cons_cbp.fipstate == f) & (cons_cbp.NAICS_3D == n)
                        ].index

                    if  c_state_counts.loc[(f, n), c] == 0:
                        pass

                    else: 
                        cons_cbp.loc[frac_index, c + '_%'] = \
                            cons_cbp.loc[frac_index, c] / \
                                c_state_counts.loc[(f, n), c]

        return cons_cbp

    @staticmethod   
    def county_energy_calc(state_fuel_MMBtu, cons_cbp):
        """
        Calculate county-level energy use (in TBtu) for construction NAICS. 
        """
        cons_county_energy = pd.DataFrame(
            cons_cbp[['fipstate', 'fipscty', 'naics']]
            )

        for c in ['fipstate', 'naics']:
            cons_county_energy.loc[:, c] = [
                int(x) for x in cons_county_energy[c]
                ]

        for fuel in state_fuel_MMBtu.Fuel_type.drop_duplicates():
            for fips in state_fuel_MMBtu.State_FIPS.drop_duplicates():
                for n in state_fuel_MMBtu.NAICS_3D.drop_duplicates():
                    state_energy = state_fuel_MMBtu[
                        (state_fuel_MMBtu.Fuel_type == fuel) & \
                        (state_fuel_MMBtu.State_FIPS == fips) & \
                        (state_fuel_MMBtu.NAICS_3D == n)
                        ].loc[:, ('n1_4'):('n1000')].values

                    cty_index = cons_cbp[(cons_cbp.fipstate == fips) & \
                        (cons_cbp.NAICS_3D == int(n))].index

                    cty_energy = state_energy * \
                        cons_cbp.loc[cty_index, ('n1_4_%'):('n1000_%')]
                        
                    cons_county_energy.loc[cty_index, fuel] = \
                        cty_energy.sum(axis=1) / 1e6

        return cons_county_energy
