import pandas as pd

import numpy as np

class Mining(object):
    """
    Methods for calculating count-level energy data for mining sector.
    """

    @staticmethod
    def national_data(mining_file):
        """Import 2012 Economic Census data for mining. Fuel use in MMBtu.
        """
        data = pd.read_excel(mining_file, \
            sheetname = 'Mining_6D_US_MMBtu', index_col = 'NAICS_2012')

        data.loc[:,'Other'] = data.Misc + data.Crude

        data.drop(
            ['fac_count_2012', 'val_ship_dollars', 'Misc', 'Crude'], \
                axis = 1, inplace = True
            )

        national_2014_TBtu = data.multiply(
            data.Production_growth/1E6, axis = 'index'
            )

        national_2014_TBtu.drop(['Production_growth'], axis = 1, inplace = True)

        return national_2014_TBtu

    @staticmethod       
    def county_frac_calc(cbp_corrected):
        """
        Apply mining intensities by fuel type and 6-digit NAICS codes 
        to corrected 2014 CBP facility counts. Does not included
        electricity from mining facilities reporting on Form EIA-923.
        """
        mining_cbp = cbp_corrected[(cbp_corrected.naics > 210000) & \
            (cbp_corrected.naics < 220000)][
                ['fipstate', 'fipscty', 'naics', 'est', 'n1_4', 'n5_9', 
                    'n10_19','n20_49', 'n50_99', 'n100_249', 'n250_499', 
                        'n500_999','n1000']
                ]

        m_national_counts = mining_cbp.groupby('naics')[
            ['n1_4', 'n5_9', 'n10_19','n20_49', 'n50_99', 'n100_249', 
                'n250_499', 'n500_999','n1000']
            ].aggregate(np.sum).sum(axis = 1)

        mining_cbp.loc[:, 'natl_fraction'] = mining_cbp.est.divide(
            mining_cbp.naics.map(m_national_counts))

        return mining_cbp

    @staticmethod   
    def county_energy_calc(mining_cbp, national_2014_TBtu, GHGs):
        """
        Calculate county-level energy use (in TBtu) for mining NAICS. 
        """
        county_energy = pd.DataFrame(
            mining_cbp[['fipstate', 'fipscty', 'naics']]
            )

        county_energy.loc[:, 'fipstate'] = [
            int(x) for x in county_energy.fipstate
            ]

        for n in national_2014_TBtu.index:
            for fuel in national_2014_TBtu.columns:
                m_index = mining_cbp[mining_cbp.naics == n].index

                county_energy.loc[m_index, fuel] = \
                    mining_cbp.loc[m_index, 'natl_fraction'] * \
                        national_2014_TBtu.loc[n, fuel]


        return county_energy    

    #to run:
    #mining_national_2014 = mining.national_data()
    #minining_cbp = mining.county_frac_calc()
    #mining_county_energy = mining.county_energy_calc(mining_cbp, mining_national_2014)
