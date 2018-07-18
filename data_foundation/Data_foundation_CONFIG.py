import numpy as np
import pandas as pd
import os
import itertools as itools
import datetime

os.chdir("Y:/6A20/Public/ICET/Code/DataFoundation/")
from county_calculations.Calculate_MfgEnergy import *
from county_calculations.Match_GHGRP_County import *
from county_calculations.Match_GHGRP_EIA_CHP import *
from county_calculations.Calculate_Construction import *
from county_calculations.Calculate_Ag import *
from county_calculations.Calculate_Mining import *
from county_calculations.Calculate_All_Industry import *
from county_calculations.MatchMECS_NAICS import *

# Files
working_dir = "Y:/6A20/Public/ICET/Data for calculations/Data foundation/"
ag_file = working_dir + "Ag_Model_Inputs.xlsx"
farm_counts_file = working_dir + "USDA_Census_Farm-counts.csv"
mining_file = working_dir + "Mining_2012Census_IPF_input-calc.xlsx"
construction_file = working_dir + "Cons_State_2012Census_InputCalc.xlsx"
ghgrp_energy = working_dir + "GHGRP_all_20170908-1246.csv"
MECS_IPF_results = working_dir + "naics_employment.csv"

os.chdir(working_dir)

# Import energy data calculated from GHGRP
GHGs = pd.read_csv(ghgrp_energy, encoding='latin_1')

GHGs.reset_index(inplace=True, drop=True)

# Correct PRIMARY_NAICS_CODE from 561210 to 324110 for Sunoco, Inc. Toledo
# Refinergy (FACILITY_ID == 1001056).
fix_dict = {
    'PRIMARY_NAICS_CODE': 324110, 'PNC_3': 324,
    'GROUPING': 'Petroleum and Coal Products'
    }

for k, v in fix_dict.items():
    GHGs.loc[GHGs[GHGs.FACILITY_ID == 1001056].index, k] = v

empsize_dict = {
    'Under 50': 'n1_49', '50-99': 'n50_99', '100-249': 'n100_249',
    '250-499': 'n250_499', '500-999': 'n500_999',
    '1000 and Over': 'n1000'
    }

# Begin calculations of county-level data.
County_matching.year = 2010

cbp_for_matching_2010 = County_matching.cbp_data(working_dir)

ghgrp_for_matching_2010 = County_matching.ghgrp_data(
    GHGs, cbp_for_matching_2010
    )

cbp_for_matching_2010 = County_matching.ghgrp_counts(
    cbp_for_matching_2010, ghgrp_for_matching_2010, working_dir
    )

cbp_corrected_2010 = County_matching.cbp_corrected_calc(cbp_for_matching_2010)

Manufacturing_energy.year = 2010

Manufacturing_energy.empsize_dict = empsize_dict

Manufacturing_energy.update_naics(GHGs, ghgrp_for_matching_2010)


#MatchMECS_NAICS(GHGs, 'NAICS_USED')

IPF_MECS_formatted = Manufacturing_energy.format_IPF(MECS_IPF_results)

MECS_intensities = Manufacturing_energy.calc_intensities(
    IPF_MECS_formatted, cbp_for_matching_2010)

# Run "county matching" first for 2010 and then for 2014.
# In between, need to run all of "Calculate CountyEnergy", with 2010 cbp
# and then the final part with 2014 cbp data.
County_matching.year = 2014

cbp_for_matching_2014 = County_matching.cbp_data(working_dir)

ghgrp_for_matching_2014 = County_matching.ghgrp_data(
    GHGs, cbp_for_matching_2014
    )

# Export dataframe as csv to use in end use calculations.
ghgrp_for_matching_2014.to_csv('ghgrp2014_NAICS_matched.csv')

cbp_for_matching_2014 = County_matching.ghgrp_counts(
    cbp_for_matching_2014, ghgrp_for_matching_2014, working_dir
    )

cbp_corrected_2014 = County_matching.cbp_corrected_calc(cbp_for_matching_2014)

Manufacturing_energy.year = 2014

Manufacturing_energy.update_naics(GHGs, ghgrp_for_matching_2014)

# Run flag_counties method to ID counties where GHGRP NAICS and CBP NAICS
# don't align. Exports 'flagged_county_list.csv' to working directory.
# flagged_counties = County_matching.flag_counties(
#     cbp_for_matching_2014, ghgrp_for_matching_2014
#     )

GHGRP_electricity = format_eia923(GHGs)

cbp_corrected_923 = Manufacturing_energy.GHGRP_electricity_calc(
    GHGRP_electricity, cbp_for_matching_2014
    )

mfg_county_energy = Manufacturing_energy.combfuel_calc(
    cbp_corrected_2014, MECS_intensities
    )

# This method incorporates mining operations that are covered by
# the GHGRP.
mfg_county_energy = Manufacturing_energy.final_merging(
    ghgrp_for_matching_2014, GHGs, mfg_county_energy
    )

# Calculate electricity use in manufacturing and GHGRP facilities that report
# with Form EIA-923.
mfg_county_energy = Manufacturing_energy.elec_calc(
    GHGRP_electricity, mfg_county_energy, cbp_corrected_923, MECS_intensities
    )

# Execute the construction methods to calculate county-level 
# construction energy use.
cons_input = Cons.cons_dict(construction_file)

cons_state_energy = Cons.state_energy_calc(cons_input)

cons_cbp = Cons.county_frac_calc(cbp_corrected_2014)

cons_county_energy = Cons.county_energy_calc(cons_state_energy, cons_cbp)

# Execute the ag methods to calculate county-level agriculture energy use.
ag_county_counts = Ag.county_counts_calc(farm_counts_file)

ag_county_energy = Ag.county_energy_calc(ag_county_counts, ag_file)


# Execute the mining methods to calculate couty-level mining eneryg use.
mining_national_2014 = Mining.national_data(mining_file)

mining_cbp = Mining.county_frac_calc(cbp_corrected_2014)

mining_county_energy = Mining.county_energy_calc(
    mining_cbp, mining_national_2014, GHGs
    )

# Reformat nonmanufacturing sector dataframes to match manufacturing sector
# dataframe
Industry_merge.fix_columns(
    ag_county_energy, cons_county_energy, mining_county_energy
    )

for nmfg in [ag_county_energy, cons_county_energy, mining_county_energy]:
    Industry_merge.county_index(nmfg)


# Merge and format sector dataframes
county_energy = Industry_merge.energy_calc(
    ag_county_energy, cons_county_energy, mining_county_energy,
    mfg_county_energy)

# Add timestamp to output csv file.
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
filename = 'County_IndustryDataFoundation_2014_update_' + ts + '.csv'

county_energy.to_csv(filename)

# Results summary
filename_summary = '2014_Summaries_' + ts + '.xlsx'
with pd.ExcelWriter(filename_summary) as writer:
    county_energy.groupby('fips_matching')[Industry_merge.MECS_FT].sum(
        ).to_excel(writer, sheet_name = 'County and Fuel Type')
    county_energy.groupby('fipstate')[Industry_merge.MECS_FT].sum().to_excel(
        writer, sheet_name = 'State and Fuel Type'
        )
    county_energy.groupby('subsector')[Industry_merge.MECS_FT].sum().to_excel(
        writer, sheet_name = 'Subsector and Fuel Type'
        )
