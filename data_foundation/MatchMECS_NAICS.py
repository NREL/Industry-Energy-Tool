
import pandas as pd

import numpy as np


def MatchMECS_NAICS(DF, naics_column):
    """
    Method for matching 6-digit NAICS codes with adjusted
     MECS NAICS.
    """

    DF[naics_column].fillna(0, inplace = True)

    DF.loc[:, naics_column] =[np.int(x) for x in 
        DF[naics_column]
        ]

    DF_index = DF[
        (DF[naics_column] > 310000) & \
        (DF[naics_column] < 400000)
        ].index
    
    nctest = [
        DF.loc[DF_index, naics_column].dropna().apply(
            lambda x: int(str(x)[
                0:len(str(x))- i
            ])) for i in range(0,4)
        ]

    nctest = pd.concat(nctest, axis = 1)

    nctest.columns = ['N6', 'N5', 'N4', 'N3']

    #Match GHGRP NAICS to highest-level MECS NAICS. Will match to 
    #"dummy" "-09" NAICS where available. This is messy, but 
    #functional.
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