import pandas as pd

import numpy as np

import itertools as itools


class County_matching(object):
    """
    Class containing methods to import and format Census Business Patterns
    (CBP) establishment count data by NAICS and employment size. Then corrects
    establishment count data using EPA GHGRP data.
    """

    #Set analysis year
    def __init__(self, year):
        self.year = 2010

    @classmethod
    def cbp_data(cls, file_dir):
        """
        Import and format Census County Business Patterns data.
        """

        cbp = pd.read_csv(
            file_dir + 'cbp' + str(cls.year)[2:4] + 'co.txt', sep=",",
            header=0, dtype = {'fipscty': np.str}
            )

        cbp.naics = cbp.naics.apply(lambda x: x.replace('-', ''))

        cbp.naics = cbp.naics.apply(lambda x: x.replace('/', ''))

        cbp.loc[cbp[cbp.naics == ''].index, 'naics'] = 0

        cbp.naics = cbp.naics.astype('int')

        cbp['naics_n'] = [len(str(n)) for n in cbp.naics]

        cbp['industry'] = cbp.loc[cbp[cbp.naics != 0].index, 'naics'].apply(
            lambda x: int(str(x)[0:2]) in [11, 21, 23, 31, 32, 33]
            )

        cbp = pd.DataFrame(cbp[cbp.industry == True])

        cbp['fips_matching'] = cbp.fipstate.apply(lambda x: str(x)) + \
            cbp.fipscty

        cbp['fips_matching'] = cbp.fips_matching.apply(lambda x: int(x))

        #Correct instances where CBP NAICS are wrong
        #Hancock County, WV has a large electroplaing and rolling facility
        #that shouldn't be classified as 331110/331111 
        if cls.year == 2014:

            cbp.drop(
                cbp[(cbp.fips_matching == 54029) &
                (cbp.naics == 331110)].index, inplace=True
                )

        else:
            cbp.drop(
                cbp[(cbp.fips_matching == 54029) &
                (cbp.naics == 331111)].index, inplace=True
                )

        cbp_for_matching = pd.DataFrame(cbp[(cbp.naics_n == 6) & \
                (cbp.industry == True)
                ]
            )

        #Create n1-49 column to match MECS reporting.
        cbp_for_matching.loc[:, 'n1_49'] = cbp_for_matching[[
            'n1_4', 'n5_9', 'n10_19', 'n20_49'
            ]].sum(axis=1)


        cbp_for_matching['fips_n'] = [
            i for i in zip(cbp_for_matching.loc[:, 'fips_matching'],
                cbp_for_matching.loc[:,'naics'])
            ]

        #Remove state-wide "999" county FIPS
        cbp_for_matching = pd.DataFrame(
            cbp_for_matching[cbp_for_matching.fipscty != '999']
            )

        return cbp_for_matching

    @classmethod    
    def ghgrp_data(cls, GHGs, cbp_for_matching):
        """
        Import GHGRP reporting data and calculated energy. Create count of GHGRP 
        reporters by zip and NAICS.Excludes facilities where calculated
        energy == 0. This avoids removing facilities from the CBP count that
        don't have a GHGRP-calculated energy value.
        """

        ghgrp = pd.DataFrame(
            GHGs[(GHGs.REPORTING_YEAR == cls.year) & (GHGs.MMBtu_TOTAL !=0)],
                columns = [
                    'FACILITY_ID', 'FACILITY_NAME', 'FUEL_TYPE', 'UNIT_NAME',
                    'COUNTY', 'COUNTY_FIPS', 'LATITUDE', 'LONGITUDE', 'STATE',
                    'ZIP', 'PRIMARY_NAICS_CODE', 'SECONDARY_NAICS_CODE',
                    'MECS_Region', 'MMBtu_TOTAL'
                    ]
            )

        # Apply primary NAICS code as secondary NAICS code where facility has 
        # reported none.
        ghgrp.loc[
            ghgrp[ghgrp.SECONDARY_NAICS_CODE.isnull()==True].index, 
            'SECONDARY_NAICS_CODE'
                ] = ghgrp.loc[
                    ghgrp[ghgrp.SECONDARY_NAICS_CODE.isnull() == True].index,
                    'PRIMARY_NAICS_CODE'
                    ].values

        ghgrp.COUNTY_FIPS.fillna(0, inplace = True)

        for c in ['SECONDARY_NAICS_CODE', 'PRIMARY_NAICS_CODE', 'COUNTY_FIPS']:

            ghgrp.loc[:, c] = ghgrp[c].astype('int')

        ghgrp['INDUSTRY'] = ghgrp.PRIMARY_NAICS_CODE.apply(
            lambda x: (int(str(x)[0:2]) in [11, 21, 23, 31, 32, 33])
                ) | (
                    ghgrp.SECONDARY_NAICS_CODE.apply(
                            lambda x: (int(str(x)[0:2]
                            ) in \
                        [11, 21, 23, 31, 32, 33])
                        )
                    )

        ghgrp_for_matching = pd.DataFrame(
            ghgrp[ghgrp.INDUSTRY == True].drop_duplicates(['FACILITY_ID']).loc[
                :, ('COUNTY_FIPS', 'FACILITY_ID', 'FACILITY_NAME', \
                        'PRIMARY_NAICS_CODE', 'SECONDARY_NAICS_CODE'
                    )
                ]
            )

        # Update NAICS (2007) to 2012 NAICS if year > 2011
        if cls.year > 2011:
            naics07_12 = pd.read_excel('2007_to_2012_NAICS.xls', skiprows=2,
                parse_cols=3
                )

            for c in ['2007 NAICS Code', '2012 NAICS Code']:

                naics07_12.loc[:, c] = naics07_12[c].astype('int32')

            naics07_12.drop_duplicates(
                '2007 NAICS Code', keep='last', inplace =True
                )

            naics07_12.set_index('2007 NAICS Code', inplace = True)


            for c in ['PRIMARY_NAICS_CODE', 'SECONDARY_NAICS_CODE']:

                ghgrp_for_matching[c +'_12'] = [
                    naics07_12.ix[[n]]['2012 NAICS Code'].values[0] for n in \
                        ghgrp_for_matching[c]
                    ]

                ghgrp_for_matching['FIP_' + c[0] + 'N_12'] = [
                    i for i in zip(
                        ghgrp_for_matching.loc[:, 'COUNTY_FIPS'],
                            ghgrp_for_matching.loc[:, c + '_12']
                        )
                    ]

        else:
            for c in ['PRIMARY_NAICS_CODE', 'SECONDARY_NAICS_CODE']:

                ghgrp_for_matching['FIP_' + c[0] +'N'] = [
                    i for i in zip(ghgrp_for_matching.loc[:,'COUNTY_FIPS'],
                            ghgrp_for_matching.loc[:,c]
                        )
                    ]


    #Find updated GHGRP NAICS that aren't in the CBP.
    #Results show that 111419 Other Food Crops Grown Under Cover is not listed 
    #in the CBP. There are only two GHGRP facilities that report under this 
    #NAICS. It is not clear what the alternative NAICS should be based on the 
    #NAICS values reported for the counties the GHGRP facilities are located in. 

        ghgrp_manual = {}

        if cls.year > 2011:
            
            ghgrp_for_matching['pn12_in_cbp']  = [
                n in cbp_for_matching.naics.values for \
                    n in ghgrp_for_matching.PRIMARY_NAICS_CODE_12.values
                ] 

            ghgrp_for_matching['sn12_in_cbp']  = [
                n in cbp_for_matching.naics.values for \
                    n in ghgrp_for_matching.SECONDARY_NAICS_CODE_12.values
                ] 

            for k in ghgrp_for_matching[
                (ghgrp_for_matching.pn12_in_cbp == False) & 
                    (ghgrp_for_matching.sn12_in_cbp == False)
                ]['PRIMARY_NAICS_CODE_12']:ghgrp_manual[k] = 0

        else:
            ghgrp_for_matching['pn_in_cbp']  = [
                n in cbp_for_matching.naics.values for \
                    n in ghgrp_for_matching.PRIMARY_NAICS_CODE.values
                ] 

            ghgrp_for_matching['sn_in_cbp']  = [
                n in cbp_for_matching.naics.values for \
                    n in ghgrp_for_matching.SECONDARY_NAICS_CODE.values
                    ] 

            for k in ghgrp_for_matching[
                (ghgrp_for_matching.pn_in_cbp == False) & 
                    (ghgrp_for_matching.sn_in_cbp == False)]\
                ['PRIMARY_NAICS_CODE']:
                    ghgrp_manual[k] = 0


        #Select NAICS that corresponds to CBP data.
        if cls.year > 2011:

            ghgrp_for_matching.loc[
                ghgrp_for_matching[ghgrp_for_matching.pn12_in_cbp == True].index,
                    'NAICS_USED'
                ] = ghgrp_for_matching[ghgrp_for_matching.pn12_in_cbp == True][
                    'PRIMARY_NAICS_CODE_12'
                    ]

            ghgrp_for_matching.loc[ghgrp_for_matching[(
                
                ghgrp_for_matching.pn12_in_cbp == False
                
                    ) & (ghgrp_for_matching.sn12_in_cbp == True)].index, 'NAICS_USED'
                
                ] = ghgrp_for_matching[(
                
                    ghgrp_for_matching.pn12_in_cbp == False
                
                    ) & (ghgrp_for_matching.sn12_in_cbp == True)][
                    'SECONDARY_NAICS_CODE_12'
                    ]

        else:
            
            ghgrp_for_matching.loc[

                ghgrp_for_matching[ghgrp_for_matching.pn_in_cbp == True].index,
            
                    'NAICS_USED'
            
                ] = ghgrp_for_matching[ghgrp_for_matching.pn_in_cbp == True][
            
                    'PRIMARY_NAICS_CODE'
            
                    ]

            ghgrp_for_matching.loc[ghgrp_for_matching[(
            
                ghgrp_for_matching.pn_in_cbp == False
            
                    ) & (ghgrp_for_matching.sn_in_cbp == True)].index, 'NAICS_USED'
            
                ] = ghgrp_for_matching[(
            
                    ghgrp_for_matching.pn_in_cbp == False
            
                ) & (ghgrp_for_matching.sn_in_cbp == True)]['SECONDARY_NAICS_CODE']

        return ghgrp_for_matching

    @classmethod
    def ghgrp_counts(cls, cbp_for_matching, ghgrp_for_matching, file_dir):
        """
        Method for adding GHGRP facility counts to formatted CBP data.
        Identify which NAICS codes in the Census data are covered in MECS
        Begin by importing MECS data. Note that CBP data after 2011 use 2012 
        NAICS. MECS data are based on 2007 NAICS. Most significant difference is
        aggregationof Alkalies and Chlorine Manufacturing, 
        Carbon Black Manufacturing, and 
        All other Basic Inorganic Chemical Manufacturing into a single 
        NAICS code.
        """

        cbp_for_matching['in_ghgrp'] = [
            n in ghgrp_for_matching.COUNTY_FIPS.values for n in \
                cbp_for_matching.fips_matching.values
            ]

        #Drop values not associated with county FIPS found in GHGRP reporters
        
        # cbp_ghgrp = pd.DataFrame(
        #     cbp_for_matching[cbp_for_matching.in_ghgrp == True]
        #     )


        def facility_counts(df, c):
            """
            Counts how many GHGRP-reporting facilities by NAICS are located
            in a county
            """
            counts = pd.DataFrame(df.groupby(c)['COUNTY_FIPS'].count())
            counts.columns = ['FAC_COUNT']
            return counts

        #Create dictionaries of ghgrp facility counts based on NAICS updated 
        #to 2012 values.
        if cls.year > 2011:
            ghgrpcounts_FIPPN_dict = facility_counts(ghgrp_for_matching, 
                'FIP_PN_12')['FAC_COUNT'].to_dict()

            ghgrpcounts_FIPSN_dict = facility_counts(ghgrp_for_matching, 
                'FIP_SN_12')['FAC_COUNT'].to_dict()
        else:
            ghgrpcounts_FIPPN_dict = facility_counts(ghgrp_for_matching, 
                'FIP_PN')['FAC_COUNT'].to_dict()

            ghgrpcounts_FIPSN_dict = facility_counts(ghgrp_for_matching, 
                'FIP_SN')['FAC_COUNT'].to_dict()

        #Map GHGRP facilities count to Census data
        cbp_for_matching['ghgrp_pn'] = [fn in ghgrpcounts_FIPPN_dict.keys()
            for fn in cbp_for_matching.fips_n.tolist()
            ]

        cbp_for_matching['ghgrp_sn'] = [fn in ghgrpcounts_FIPSN_dict.keys() 
            for fn in cbp_for_matching.fips_n.tolist()
            ]

        cbp_for_matching['ghgrp_fac'] = 0

        cbp_for_matching['est_small'] = cbp_for_matching.loc[
            :, ('n1_4'): ('n20_49')].sum(axis = 1)

        cbp_for_matching['est_large'] = cbp_for_matching.loc[
            :, ('n50_99'): ('n1000')].sum(axis = 1)

        if cls.year > 2011:
            
            MECS_NAICS = pd.read_csv(file_dir + 'mecs_naics_2012.csv')

        else:

            MECS_NAICS = pd.read_csv(file_dir + 'mecs_naics.csv')

        cbp_for_matching['n_in_mecs'] = [
            n in MECS_NAICS.values for n in cbp_for_matching.naics
            ]


        #Method for matching 6-digit NAICS codes with adjusted MECS NAICS.  
        def MatchMECS_NAICS(DF, naics_column):
            DF[naics_column].fillna(0, inplace = True)

            DF.loc[:, naics_column] =[np.int(x) for x in 
                DF[naics_column]
                ]

            DF_index = DF[DF[naics_column]>0].index
            
            nctest = [
                DF.loc[DF_index, naics_column].apply(lambda x: int(str(x)[
                    0:len(str(x))- i
                    ])) for i in range(0,4)
                ]

            nctest = pd.concat(nctest, axis = 1)

            nctest.columns = ['N6', 'N5', 'N4', 'N3']

            #Pare down to manufacturing NAICS only (311 - 339)
            nctest = pd.DataFrame(
                nctest[(nctest.N3 >= 311) & (nctest.N3 <= 339)]
                )

            #Match GHGRP NAICS to highest-level MECS NAICS. Will match to 
            #"dummy-09" NAICS where available. This is messy, but functional.
            ncmatch = pd.concat(
                [pd.merge(nctest, MECS_NAICS, left_on=nctest[
                    column], right_on = MECS_NAICS.MECS_NAICS, \
                        how = 'left').iloc[:,4] 
                            for column in nctest.columns], axis =1
                )

            ncmatch.index = nctest.index

            ncmatch['NAICS_MATCH'] = ncmatch.apply(
                lambda x: int(list(x.dropna())[0]), axis = 1
                )

            #Update GHGRP dataframe with matched MECS NAICS.
            DF.loc[ncmatch.index, 'MECS_NAICS'] = ncmatch.NAICS_MATCH


        # Match Census NAICS to NAICS available in MECS. Note that MECS does
        # not include agriculture, mining, and construction industries and does 
        # not include 6-digit detail for all manufacturing NAICS (31 - 33)
        MatchMECS_NAICS(cbp_for_matching, 'naics')

        MECS_regions = pd.read_csv(
            file_dir + 'US_FIPS_Codes.csv', index_col = ['COUNTY_FIPS']
            )

        cbp_for_matching.loc[:,'MECS_Region'] = cbp_for_matching.fipstate.map(
            dict(MECS_regions[['FIPS State', 'MECS_Region']].values))

        matching_index_pn = cbp_for_matching[
            (cbp_for_matching.ghgrp_pn == True)
            ].index

        matching_index_sn =  cbp_for_matching[
            (cbp_for_matching.ghgrp_sn == True) & \
                (cbp_for_matching.ghgrp_pn == False)
            ].index

        cbp_for_matching.loc[matching_index_pn, 'ghgrp_fac'] = \
            cbp_for_matching.loc[matching_index_pn, 'fips_n'].map(
                ghgrpcounts_FIPPN_dict
            )

        cbp_for_matching.loc[matching_index_sn, 'ghgrp_fac'] = \
            cbp_for_matching.loc[matching_index_sn, 'fips_n'].map(
                ghgrpcounts_FIPSN_dict
            )

        return cbp_for_matching

    @classmethod        
    def flag_counties(cls, cbp_for_matching, ghgrp_for_matching):
        """
        Identifies counties where the GHGRP or CBP NAICS designation is 
        potentially incorrect.
        Outputs to working drive csv ("flagged_county_list") of flagged 
        counties.
        """
        count_flagged_cbp = pd.DataFrame(
            cbp_for_matching[cbp_for_matching.in_ghgrp == True], copy = True
            )

        count_flagged_cbp['N2'] = count_flagged_cbp.naics.apply(
            lambda x: int(str(x)[0:2]))

        count_flagged_cbp = count_flagged_cbp[
            count_flagged_cbp.N2 != 23
            ]

        count_flagged_cbp.drop('N2', axis=1, inplace=True)

        count_flagged_cbp = count_flagged_cbp.groupby(
            ['fips_matching', 'naics']
            )['ghgrp_fac'].sum()

        if cls.year > 2011:
            ghgrp_count = ghgrp_for_matching.groupby([
                'COUNTY_FIPS', 'PRIMARY_NAICS_CODE_12'])['FACILITY_ID'].count()

        else:
            ghgrp_count = ghgrp_for_matching.groupby([
                'COUNTY_FIPS', 'PRIMARY_NAICS_CODE'])['FACILITY_ID'].count()

        fac_count_compare = pd.concat(
            [count_flagged_cbp, ghgrp_count], axis = 1
            )

        flagged_list = pd.DataFrame(
            fac_count_compare[fac_count_compare.ghgrp_fac.isnull() == True]
            )

        flagged_list.to_csv('flagged_county_list.csv')  

        return flagged_list

    # ghgrp_for_matching['flagged'] = [i in fac_count_compare[(
    #   fac_count_compare.ghgrp_fac != fac_count_compare.FACILITY_ID
    #   )].index for i in ghgrp_for_matching.COUNTY_FIPS]

    # #Export flagged facilities for manual NAICS correction
    # ghgrp_for_matching[ghgrp_for_matching.flagged == True].iloc[:,0:5].to_csv(
    #   'ghgrp_facilities_county-flagged.csv'
    #   )

    @staticmethod
    def cbp_corrected_calc(cbp_for_matching):
        """
        Method for correcting CBP facility counts based on GHGRP facilities.
        """

        # cbp_corrected = pd.DataFrame(
        #     cbp_for_matching.values, index = cbp_for_matching.index, \
        #         columns = list(cbp_for_matching.columns), copy = False
        #     )
        
        def fac_correct(df, index):
            """
            Removes the largest facilities in a given county with matching 
            GHGRP facilities.
            """

            large = ['n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']

            small = ['n1_4', 'n5_9', 'n10_19', 'n20_49']

            ghgrp_fac_count = df.loc[index, 'ghgrp_fac']

            fac_count_large = df.loc[index, 'est_large']

            fac_count_small = df.loc[index, 'est_small']

            fac_count_total = df.loc[index, 'est']

            if ghgrp_fac_count <= fac_count_total:

                n = ghgrp_fac_count

            else:
                n = fac_count_total

            while n > 0:
                maxsize = [c for c in itools.compress(small + large, df.ix[
                        index, ('n1_4'):('n1000')
                    ].values)][-1]

                df.loc[index, maxsize] = df.loc[index, maxsize] - 1

                n = n - 1

            df.loc[index, 'est_large_corrected'] = df.loc[
                index, ('n50_99'):('n1000')
                ].sum()

            df.loc[index, 'est_small_corrected'] = df.loc[
                index, ('n1_4'):('n20_49')
                ].sum()

        # else:
        #   pass

        # if (ghgrp_fac_count > 0) & (fac_count_large !=0):

        #   if ghgrp_fac_count > fac_count_large:

        #       n = ghgrp_fac_count

        #       while n > 0:

        #           maxsize = [c for c in itools.compress(small + large, df.ix[
        #                   index, ('n1_4'):('n1000')
        #               ].values)][-1]

        #           df.loc[index, maxsize] = df.loc[index, maxsize] - 1

        #           n = n - 1
            
        #   df.loc[index, 'est_large_corrected'] = df.loc[
        #       index, ('n50_99'):('n1000')
        #       ].sum()

        #   df.loc[index, 'est_small_corrected'] = df.loc[
        #       index, ('n1_4'):('n20_49')
        #       ].sum()


        #   if ghgrp_fac_count <= fac_count_large:
                
        #       n = ghgrp_fac_count

        #       while n > 0:        

        #           maxsize = [c for c in itools.compress(large, df.ix[index, 
        #               ('n50_99'):('n1000')].values)][-1]
                    
        #           df.loc[index, maxsize] = df.loc[index, maxsize] - 1

        #           n = n - 1

        #   df.loc[index, 'est_large_corrected'] = df.loc[
        #       index, ('n50_99'):('n1000')
        #       ].sum()

        # if (ghgrp_fac_count > 0) & (fac_count_large == 0):

        #       if ghgrp_fac_count <= fac_count_small:

        #           n = ghgrp_fac_count
                
        #       else: 

        #           n = fac_count_small

        #       while n > 0:    

        #           maxsize = [c for c in itertools.compress(
        #               small, cbp_corrected.ix[index, ('n1_4'):('n20_49')].values
        #               )][-1]

        #           df.loc[index,maxsize] = df.loc[index, maxsize] - 1

        #           n = n - 1

        #       df.loc[index, 'est_small_corrected'] = df.loc[
        #           index, ('n1_4'):('n20_49')
        #           ].sum()

        # else:
        #   pass

        #Apply method for removing GHGRP facilities from the counts of the 
        #largest CBP facilities. 'cbp_for_matching' contains the original 
        #CBP facility counts. 
        cbp_ghgrp = pd.DataFrame(
            cbp_for_matching[cbp_for_matching.in_ghgrp == True], copy=True
            )

        for i in cbp_ghgrp[cbp_ghgrp.ghgrp_fac > 0].index:
            
            fac_correct(cbp_ghgrp, i)

        cbp_corrected = pd.DataFrame(cbp_for_matching, copy=True)

        cbp_corrected.update(cbp_ghgrp)

        return cbp_corrected


        #The following data frame provides the original facility counts for 
        #the matching counties provided in cbp_corrected.
        #Aggregate original and corrected CBP manufacturing facility counts 
        #by MECS region.
        # cbp_original_byMECS = cbp_for_matching[
        #     cbp_for_matching.MECS_NAICS != 0
        #     ].groupby(['MECS_Region', 'MECS_NAICS'])

        # cbp_corrected_byMECS = cbp_corrected[
        #     cbp_corrected.MECS_NAICS != 0
        #     ].groupby(['MECS_Region', 'MECS_NAICS'])
    
