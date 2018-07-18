import pandas as pd

import re


def format_eia923(ghgrp):
    """
    Iport EIA 923 data for industrial cogen and non-cogen facilities and match
    to GHGRP facilities for county-level energy calculations. 
    Correct fuel consumption for these facilities to avoid double counting. 
    """

    #EIA923 data in 'disposition' and 'gen_fuel' are imported as separate 
    #DataFrames in the eia923 dictionary.
    eia923 = pd.read_excel("923data.xlsx", sheetname=None,
        convert_float=True)

    # The EPA GHGRP crosswalk between EIA Plant ID and GHGRP Facility ID is 
    # incomplete. The necessary crosswalk is created with the following:

    # fix Riker's Island encoding issues
    ghgrp.loc[ghgrp.FACILITY_ID == 1009700, 'FACILITY_NAME'] = \
            "NYC-DOC Riker's Island"

    eia_names = eia923["disposition"][['Plant_Name', 
        'Plant_ID', 'Plant_State']].drop_duplicates()

    epa_names = pd.DataFrame(ghgrp[['FACILITY_NAME', 
        'FACILITY_ID', 'STATE']].drop_duplicates())

    eia_names_bystate = eia_names.groupby('Plant_State')

    epa_names_bystate = epa_names.groupby('STATE')

    xwalk = pd.DataFrame()


    #The following reg ex is only partially successful at creating the 
    #crosswalk. Most of the crosswalk was performed manually and is 
    #imported below.
    def epa_eia_match_names(name, state):

        pattern = re.compile(name.replace(" ", "|"))

        #Plant_ID = eia_names[eia_names.Plant_Name == name]['Plant_ID']

        epa_subset = epa_names_bystate.get_group(state)

        found = epa_subset.loc[epa_subset.FACILITY_NAME.apply(
            lambda x: pattern.search(x, re.I)
                ).dropna().index, ("FACILITY_NAME", "FACILITY_ID")]

        found['EIA_name'] = name

        #found['EIA_plant_ID'] = Plant_ID

        return found

    for s in eia_names.Plant_State.drop_duplicates():

        for n in eia_names_bystate.get_group(s)['Plant_Name']:

            xwalk = xwalk.append(epa_eia_match_names(n, s))

    #Import final crosswalk between EIA923 industrial gen and cogen facilities
    # and GHGRP facilities.
    xwalk923 = pd.read_csv("eia_epa_xwalk.csv", encoding = 'latin-1')

    #Import crosswalk between EIA923 fuels and GHGRP fuels.
    xwalk923_fuels = pd.read_csv(
        "FuelsXwalk_923-GHGRP.csv", encoding='latin-1'
        )  

    #Create DataFrame from EIA923 data of GHGRP facility incoming electricity in 
    #2014.
    GHGRP_electricity = pd.DataFrame(eia923["disposition"].query(
        'In_GHGRP == ["Y"] & YEAR < [2015]')[['Plant_ID', 'Plant_State', 
            'Incoming_MWh', 'YEAR', 'Net_Use_MWh']
            ]
        )

    GHGRP_electricity.loc[:,'FACILITY_ID'] = GHGRP_electricity.Plant_ID.map(
        dict(xwalk923[['EIA_PLANT_ID', 'FACILITY_ID']].values))


    GHGRP_electricity = pd.DataFrame(
        GHGRP_electricity[GHGRP_electricity.YEAR == 2014], copy=False
        )

    for c in ['MECS_Region', 'PRIMARY_NAICS_CODE', \
        'NAICS_USED', 'COUNTY_FIPS']:
            GHGRP_electricity.loc[:, c] = GHGRP_electricity.FACILITY_ID.map(
                dict(ghgrp[['FACILITY_ID', c]].values))


    #Drop electricity data for GHGRP facilities that aren't manufacturers, 
    #i.e. do not appear in MECS.
    #GHGRP_electricity.dropna(subset = ['MECS_NAICS'], inplace = True)

    #Sum EIA923 net electricity data (in MWh; convert to TBtu) 
    #by MECS Region and MECS NAICS.
    GHGRP_electricity.loc[:, 'Net_electricity'] = GHGRP_electricity.loc[
        :, 'Net_Use_MWh'
        ] * 1000 * 3412 / 1E12


    GHGRP_electricity.dropna(subset = ['COUNTY_FIPS'], axis=0, \
        inplace=True
        )

    for c in ['COUNTY_FIPS', 'NAICS_USED']:
        GHGRP_electricity.loc[:, c].fillna(0, inplace = True)
        GHGRP_electricity.loc[:, c] = [
            int(v) for v in GHGRP_electricity[c].values
            ]

    GHGRP_electricity.loc[
       GHGRP_electricity[GHGRP_electricity.NAICS_USED == 0].index, \
           'NAICS_USED'] = GHGRP_electricity.PRIMARY_NAICS_CODE

    GHGRP_electricity["FIPS_NAICS"] = [z for z in zip(
            GHGRP_electricity.COUNTY_FIPS.values, \
                GHGRP_electricity.NAICS_USED.values
            )
        ]

    return GHGRP_electricity

