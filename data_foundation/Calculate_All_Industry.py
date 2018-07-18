import pandas as pd


class Industry_merge(object):
    """
    Merges calculated county-level 2014 agriculture, construction, and mining 
    energy data with calculated county-level 2014 manufacturing energy data.
    """
    MECS_FT = [
        'Residual_fuel_oil', 'Diesel', 'Natural_gas', 'LPG_NGL', 'Coal', \
        'Coke_and_breeze', 'Other','Net_electricity', 'Total'
        ]

    @staticmethod
    def fix_columns(ag, cons, mining):

        ag.rename(
            columns={'Electricity': 'Net_electricity', 'DIESEL': 'Diesel',
            'LP GAS': 'LPG_NGL','OTHER': 'Other', 'NAICS': 'naics'},
            inplace=True
            )

        ag.loc[:, 'Other'] = ag.Other +  ag.GASOLINE

        for c in ['GASOLINE', 'StateFraction']:
            ag.drop(c, axis=1,inplace=True)

        cons.rename(
            columns={'Electricity': 'Net_electricity', 'LPG': 'LPG_NGL'},
            inplace=True
            )

        mining.rename(
            columns={'Electricity': 'Net_electricity'}, inplace=True
            )

        mining.loc[:, 'Other'] = \
            mining.loc[:, 'Other'] + mining.loc[:, 'Gasoline']

        mining.drop('Gasoline', axis=1, inplace=True)

    @staticmethod
    def county_index(df):
        """
        Create index to match manufacturing county energy data index (a 
        concatentation of county and state FIPS and NAICS).
        """
        if 'State ANSI' in df.columns:

            for c in ['County ANSI', 'State ANSI']:
                df.loc[:, c] = df[c].apply(lambda x: str(x))

            for i in df['County ANSI'].index:
                if len(df.loc[i, 'County ANSI']) == 1:
                    df.loc[i, 'County ANSI'] = \
                        '00' + df.loc[i, 'County ANSI']

                if len(df.loc[i, 'County ANSI']) == 2:
                    df.loc[i, 'County ANSI'] = \
                        '0' + df.loc[i, 'County ANSI']

            FIPS_STATE = \
                df['State ANSI'].values + df['County ANSI'].values
                
            FIPS_STATE = [int(x) for x in FIPS_STATE]

            for c in ['County ANSI', 'State ANSI']:
                df.loc[:, c] = df[c].apply(lambda x: int(x))

            df['fips_matching'] = FIPS_STATE

            revised_NAICS = df.loc[:, 'naics']

            NAICS_revisions = {
                '11193 & 11194 & 11199': '1119', '1125 & 1129': '1127'
                }

            for k in NAICS_revisions:
                revised_NAICS.replace(
                    to_replace=k, value=NAICS_revisions[k], inplace=True
                    )

            df['FIPS_NAICS'] = [
                z for z in zip(
                    FIPS_STATE, revised_NAICS.apply(lambda x: int(x))
                    )
                ]

            df.rename(
                columns={'State ANSI': 'fipstate', 'County ANSI': 'fipscty'},
                    inplace=True
                )

        else:
            state_str = df.fipstate.apply(lambda x: str(x))

            FIPS_STATE = \
                state_str + df.fipscty.values
                
            FIPS_STATE = [int(x) for x in FIPS_STATE]

            df['FIPS_NAICS'] = [
                z for z in zip(
                    FIPS_STATE, df.naics.apply(lambda x: int(x))
                    )
                ]

            df['fips_matching'] = \
                df.fipstate.apply(lambda x: str(x)) + df.fipscty

            df['fips_matching'] = df.fips_matching.apply(lambda x: int(x))

        df.set_index('FIPS_NAICS', drop=True, inplace=True)

    @classmethod
    def energy_calc(cls, ag, cons, mining, mfg):

        county_energy = pd.DataFrame()

        for sector in [ag, cons, mining, mfg]:
            sector.loc[:, 'fipstate'] = [
                int(x) for x in sector.fipstate
                ]

            sector.loc[:, 'subsector'] = sector.naics.apply(
                lambda x: int(str(x)[0:2])
                )

            county_energy = county_energy.append(sector)

        county_energy.loc[:, 'Total'] = \
            county_energy.loc[:, cls.MECS_FT].sum(axis=1)

        # county_energy.drop('FIPS_NAICS', axis = 1, inplace = True)

        #Need to reindex mining data from the two calculation approaches
        mining_all = pd.DataFrame(county_energy[county_energy.subsector == 21])

        mining_all.reset_index(inplace=True)
        
        mining_all.rename(columns={'index':'FIPS_NAICS'}, inplace=True)

        mining_all = pd.DataFrame(mining_all.groupby('FIPS_NAICS').sum())
        
        mining_all.loc[:, 'subsector'] = 21

        mining_all.loc[:, 'fips_matching'] = [x[0] for x in mining_all.index]

        mining_all.loc[:, 'naics'] = [int(x[1]) for x in mining_all.index]

        mining_all.loc[:, 'fipscty'] = \
            [str(x)[2:] for x in mining_all.fips_matching]

        mining_all.loc[:, 'fipstate'] = [
            int(str(x)[0:(len(str(x))-3)]) for x in mining_all.fips_matching
            ]

        county_energy = pd.DataFrame(
            county_energy[county_energy.subsector != 21], copy=False
            )

        county_energy = pd.concat([county_energy, mining_all])

        county_energy.set_index(['fips_matching', 'naics'], inplace=True,
            drop=False
            )

        # county_energy.loc[:, 'fips_ctystate'] = county_energy.FIPS_NAICS.apply(
        #     lambda x: int(x[0])
        #     )

        return county_energy
