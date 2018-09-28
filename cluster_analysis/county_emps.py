import pandas as pd

import numpy as np

class County_IndEmp(object):

    """
    Method for calculating the county-level industrial employment for a 
    specified NAICS level (n) in 2014.
    Method pulls in 2012 Ag Census data, which includes NAICS codes not covered
    by Census County Business Patterns data set.
    """    
    
    def __init__(self):
        datadir = 'Y:/6A20/Public/ICET/Data for calculations/'
        cbp_file = 'Data foundation/cbp14co.txt'
        ag_file = 'USDA_Census_Workers_county.csv'
        
        self.ag_est_file = \
            'U:/ICEP/Cluster analysis paper/USDA_TotFarmCounts.csv'

        self.cbp = pd.read_csv(datadir + cbp_file, sep=",", header=0, 
                          dtype={'fipscty': np.str}
            )

        self.cbp = pd.DataFrame(self.cbp[self.cbp.fipscty != '999'])

        self.cbp.naics = self.cbp.naics.apply(lambda x: x.replace('-', ''))

        self.cbp.naics = self.cbp.naics.apply(lambda x: x.replace('/', ''))

        self.cbp.loc[self.cbp[self.cbp.naics == ''].index, 'naics'] = 0

        self.cbp.naics = self.cbp.naics.astype('int')

        self.cbp['naics_n'] = [len(str(n)) for n in self.cbp.naics]

        self.cbp['industry'] = \
            self.cbp.loc[self.cbp[self.cbp.naics != 0].index, 'naics'].apply(
                lambda x: int(str(x)[0:2]) in [11, 21, 23, 31, 32, 33]
                )

        self.cbp = pd.DataFrame(self.cbp[self.cbp.industry == True], copy=True)
        
        self.cbp.loc[:, 'subsector'] = self.cbp.naics.apply(
            lambda x: int(str(x)[0:2])
            )
    
        self.cbp['fips_matching'] = self.cbp.fipstate.apply(
            lambda x: str(x)
            ) + self.cbp.fipscty
            

        self.cbp['fips_matching'] = self.cbp.fips_matching.apply(
            lambda x: int(x)
            )
        
        empflag_mp = {
            'A': 19/2, 'B': (99-20)/2, 'C': (249 - 100)/2, 'E': (499-250)/2, 
            'F': (999-500)/2, 'G': (2499-1000)/2, 'H': (4999-2500)/2, 
            'I': (9999-5000)/2, 'J': (24999-10000)/2, 'K': (49999-25000)/2, 
            'L':(99999-50000)/2, 'M': 100000 
            }

        empclass_mp = {
            'n1_4': (4-1)/2, 'n5_9': (9-5)/2, 'n10_19': (19-10)/2, 
            'n20_49': (49-20)/2, 'n50_99': (99-50)/2, 'n100_249': (249-100)/2, 
            'n250_499': (499-250)/2, 'n500_999': (999-500)/2, 'n1000': 1000
            }

        def round_dictionary(d):
            for k in d.keys():
                d[k] = np.round(d[k])

        for d in [empflag_mp, empclass_mp]:
            round_dictionary(d)

        # Fill in missing employment data.
        self.cbp.loc[self.cbp[(self.cbp.emp == 0) & \
            (self.cbp.empflag.isnull())].index, 'emp'] = \
                self.cbp[(self.cbp.emp == 0) & \
                (self.cbp.empflag.isnull())].n1_4 * empclass_mp['n1_4']

        self.cbp.loc[self.cbp.loc[:,'emp'] ==0, 'emp'] = \
            self.cbp.loc[self.cbp.loc[:,'emp'] ==0].empflag.map(empflag_mp)

        self.cbp.emp = self.cbp.emp.astype(np.int32, copy=False)

        # Add in USDA employment data from 2012 Ag Census to NAICS 11.
        # Note CBP does include data for Forestry and Logging (NAICS 113).
        # Also note some employment numbers are withheld.
        usda_emp = pd.read_csv(datadir + ag_file)

        for v in [' (D)', '(D)']:
            usda_emp.drop(
                usda_emp[usda_emp.Value == v].index, inplace=True
                )

        for c in ['Value', 'fips_matching']:
            usda_emp.loc[:, c] = usda_emp[c].astype(np.int32, copy=False)

        self.cbp.loc[self.cbp[self.cbp.naics == 11].index, 'emp'] = \
            self.cbp.loc[self.cbp[self.cbp.naics == 11].index, 'emp'].add(
                self.cbp.loc[self.cbp[self.cbp.naics == 11].index,
                    'fips_matching'].map(
                        dict(usda_emp[['fips_matching', 'Value']].values
                        )
                    ), fill_value=0
                )

    @classmethod
    def counts(cls, naics_n):
        """
        Create a dictionary of employment counts by county for agriculture
        and non-agriculture industries.
        """
        
        count_out = {}
        
        # Agriculture is treated differently in summation because USDA
        # employment counts are available only at the 2-digit level (i.e., 11)        
        count_out['non_ag'] = \
            cls.cbp[
                (cls.cbp.naics_n == naics_n) & (cls.cbp.subsector != 11)
                ].groupby(['fips_matching', 'naics']).emp.sum()
        
        count_out['ag'] = cls.cbp[cls.cbp.naics == 11].groupby(
            'fips_matching'
            ).emp.sum()
        
        return count_out
        # ind_emp_counts.to_csv( path_to_save + 'industry_totalemp_' + 'naics' + \
        #    str(naics_n) + 'county.csv'
        #   )
    
    @staticmethod
    def farm_est(ag_est_file):
        """
        Sum farm counts by county.
        """
        est_counts = pd.read_csv(ag_est_file)
            
        for c in ['County ANSI', 'State ANSI']:
            est_counts.loc[:, c] = est_counts[c].astype('str')
            
        est_counts.loc[:, 'cfips_len'] = est_counts['County ANSI'].apply(
            lambda x: len(x)
            )
            
        for i in [1, 2]:
            i_index = est_counts[est_counts.cfips_len == i].index
            
            if i == 1:
                est_counts.loc[i_index,'County ANSI'] = \
                    '00' + est_counts.loc[i_index, 'County ANSI']
            
            else:
                est_counts.loc[i_index,'County ANSI'] = \
                    '0' + est_counts.loc[i_index, 'County ANSI']   

#        for i in est_counts['County ANSI'].index:
#            if len(est_counts.loc[i, 'County ANSI']) == 1:
#                est_counts.loc[i, 'County ANSI'] = \
#                    '00' + est_counts.loc[i, 'County ANSI']
#
#            if len(est_counts.loc[i, 'County ANSI']) == 2:
#                est_counts.loc[i, 'County ANSI'] = \
#                    '0' + est_counts.loc[i, 'County ANSI']

        FIPS_STATE = est_counts['State ANSI'].values + \
                est_counts['County ANSI'].values
                
        FIPS_STATE = [int(x) for x in FIPS_STATE]

        for c in ['County ANSI', 'State ANSI']:
            est_counts.loc[:, c] = est_counts[c].astype('int')

        est_counts.loc[:, 'fips_matching'] = FIPS_STATE
        
        return est_counts.set_index('fips_matching').Value