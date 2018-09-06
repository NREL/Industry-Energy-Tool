import ghg_calcs as ghg
import stock_turnover as st
import bandwidth_IO as bandwidth
import pandas as pd
import numpy as np


# Specify directory with formatted energy end use files.
eu_dir = 'Y:\\6A20\\public\\ICET\\Code\\Tool\\Stock turnover\\'

# Formatted end use files. Energy in TBtu
regional_enduse = pd.read_csv(eu_dir + 'regional_enduse_formatted',
                              compression='gzip')

county_enduse = pd.read_csv(eu_dir + 'county_enduse_formatted',
                              compression='gzip')

# Calculate electricity emission factors
elect_efs = ghg.GHGs.elect_emission_factors()

bw = bandwidth.Bandwidth()

county_enduse, regional_enduse, bw_master_dict = \
    bw.bw_Nmatch(county_enduse, regional_enduse)                   

# Instantiate baseline projection period and stock turnover assumptions.
ST = st.Stock_Calcs(2050, 5)

# Import regional (and national) AEO industry value of shipment projections.
VS = ST.VS()

# Create dictionary for mapping variables
calc_dicts = ST.Def_Dicts()

tpc = ST.TPC_import(calc_dicts)

# Calculate new stock
stock_dict = ST.calc_stock_turnover(VS['regional'])

# Calculate energy projections. This doesn't include stock effects. Energy in 
# MMBtu
cnty_energy_proj= ST.calc_energy_proj(ST.proj_years, county_enduse,
                                         VS['growth'], geo='county')

reg_energy_proj = ST.calc_energy_proj(ST.proj_years, regional_enduse,
                                           VS['growth'], geo='regional')

reg_ghg_proj = ghg.GHGs.Calc_Proj_GHGs(elect_efs, reg_energy_proj,
                                        geo='regional', csv_exp=False)

# cnty_ghg_proj = ghg.GHGs.Calc_Proj_GHGs(elect_efs, cnty_energy_proj,
#                                         geo='county', csv_exp=True,
#                                         fname='ghg_cnty_proj')



## Next define regional unit energy consumption (uec) values for new and 
## old stock, and a weighted average.
uecs = {}

for n in ['weighted', 'new', 'old']:
    uecs[n] = ST.Calc_Baseline_uec(reg_energy_proj, county_enduse, tpc, n,
                                   detail=True)

# Baseline stock turnover scenario
uecs['weighted'] = \
  ST.Define_Stock_Char(county_enduse, reg_energy_proj,
                          stock_dict, uecs, VS, bw_master_dict
                          )

ste_reg =   \
    ST.Calc_Stock_Effects(uecs['weighted'], regional_enduse, reg_energy_proj, VS,
                          geo='regional')

ste_cnty =  \
    ST.Calc_Stock_Effects(uecs['weighted'], county_enduse, cnty_energy_proj, VS,
                          geo='county')

# # Calculate ghg emissions with stock turnover effects
# ghg_stock_reg = \
#     ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_reg,
#                                geo='regional', csv_exp=False)

# ghg_stock_cnty = \
#     ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_cnty,
#                                geo='county', csv_exp=False,
#                                fname='ghg_cnty_stock_proj')

# # Bandwidth energy efficiency scenario
# ST_ee = st.Stock_Calcs(2050, 5)

# ST_ee.set_ee_params(100, 'PM_high')

# bw_input = bandwidth.Bandwidth.set_bw_inputs(
#     bw, industry=['all'], scaling=ST_ee.ee_params['scaling'], 
#     reduction=ST_ee.ee_params['bandwidth']
#     )

# uec_weighted_ee = \
#   ST_ee.Define_Stock_Char(county_enduse, reg_energy_proj,
#                             stock_dict, uecs, VS, bw_master_dict,
#                              bw_input=bw_input)

# ste_ee_reg = \
#   ST_ee.Calc_Stock_Effects(uec_weighted_ee, regional_enduse, reg_energy_proj, VS,
#               geo='regional')

# ghg_ee_reg = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_ee_reg,
#                          geo='regional', csv_exp=False)

# ste_ee_cnty = \
#   ST_ee.Calc_Stock_Effects(uec_weighted_ee, county_enduse, cnty_energy_proj, VS,
#               geo='county')

# ghg_ee_cnty = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_ee_cnty,
#                          geo='county', csv_exp=False, fname='ghg_cnty_ee_proj')

# # Bandwidth EE and fuel switching scenario.
# ST_ee_fs = st.Stock_Calcs(2050, 5)

# ST_ee_fs.set_ee_params(100, 'PM_high')

# ST_ee_fs.set_fs_params(100, ind='All', eu=None, temp_band=None)

# # Calculate stock characteristics with ee and fs parameters
# uec_weighted_ee_fs = \
#   ST_ee_fs.Define_Stock_Char(county_enduse, reg_energy_proj,
#                              stock_dict, uecs, VS, bw_master_dict,
#                              bw_input=bw_input, fs_input=ST_ee_fs.fs_params)

# ste_ee_fs_reg = \
#   ST_ee_fs.Calc_Stock_Effects(uec_weighted_ee_fs, regional_enduse, reg_energy_proj, VS,
#               geo='regional')

# # ste_ee_fs_cnty = \
# #     ST_ee_fs.Calc_Stock_Effects(uec_weighted_ee_fs, county_enduse, cnty_energy_proj, VS,
# #               geo='county')

# ghg_ee_fs_reg = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_ee_fs_reg,
#                          geo='regional', csv_exp=False)

# # ghg_ee_fs_cnty = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_ee_fs_cnty,
# #                          geo='county', csv_exp=True, fname='ghg_cnty_ee_fs_proj')

# # Fuel switching scenario.
ST_fs = st.Stock_Calcs(2050, 5)

ST_fs.set_fs_params(100, ind='All', eu=None, temp_band=None)

# Calculate stock characteristics with ee and fs parameters
uec_weighted_fs = \
  ST_fs.Define_Stock_Char(county_enduse, reg_energy_proj,
                             stock_dict, uecs, VS, bw_master_dict,
                            fs_input=ST_fs.fs_params)

ste_fs_reg = \
  ST_fs.Calc_Stock_Effects(uec_weighted_fs, regional_enduse, reg_energy_proj, VS,
              geo='regional')

# ste_fs_cnty = \
#     ST_fs.Calc_Stock_Effects(uec_weighted_fs, county_enduse, cnty_energy_proj, VS,
#               geo='county')

# ghg_fs_reg = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_fs_reg,
#                          geo='regional', csv_exp=False)

# ghg_fs_cnty = ghg.GHGs.Calc_Proj_GHGs(elect_efs, ste_fs_cnty,
#                          geo='county', csv_exp=False, fname='ghg_cnty_fs_proj')

#Define method for exporting ghg results summary to excel
def ghg_res_xls(ghg_df, fname):
  res_writer = pd.ExcelWriter(fname + '.xlsx')
  for n in range(0, 3):
    ghg_df.sum(axis=0, level=n).sum(axis=1, level=1).to_excel(
        res_writer, 'Sheet_'+ str(n)
        )
  res_writer.save()